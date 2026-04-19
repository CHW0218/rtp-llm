# Level 2：后端入口 & Python ↔ C++ 的桥

## 🎯 本关核心心智模型

后端**不是纯 C++ 进程**，而是"**Python 进程 + 内嵌一颗 C++ 心脏**"。

分工：
- **Python 层**：一次性的事——加载权重、描述模型结构、启动服务
- **C++ 层**：反复跑的事——调度、forward、kernel、gRPC server
- **胶水**：**pybind11**（让 Python 和 C++ 互相看见对方的对象）

**一句话**：Python 爽在一次性的事，C++ 快在反复跑的事。

---

## 📍 起点：前端把 gRPC 请求发出来了，谁接？怎么到 C++？

Level 1 结束于：

```python
await self.grpc_client.post_request(...)
```

这关回答：**请求飞到后端后发生了什么？**

---

## 问题 1：后端为什么不是纯 C++？

两个原因：

**① 模型权重天然在 Python 世界里**
主流模型（Llama、Qwen、DeepSeek）都是 PyTorch 写的——权重是 `.safetensors`，加载代码是 Python。C++ 重写要兼容 HuggingFace 上千个模型，代价巨大。

**② 模型结构定义也在 Python 里**
`rtp_llm/models/llama.py`、`qwen_vl.py` 等定义"模型有几层、每层什么样"。C++ 只负责**执行**，不负责**描述**。

---

## 问题 2：Python 怎么"创建"一个 C++ 对象？—— pybind11

**pybind11** 把 C++ 类导出成 Python 能 import 的东西。

📌 `rtp_llm/cpp/pybind/multi_gpu_gpt/RtpLLMOp.cc:398`：

```cpp
void registerRtpLLMOp(const py::module& m) {
    pybind11::class_<RtpLLMOp>(m, "RtpLLMOp")    // C++ 类 → Python 类
        .def(py::init<>())                        // 导出构造函数
        .def("init", &RtpLLMOp::init, ...)        // 导出方法
        .def("stop", &RtpLLMOp::stop, ...);
}
```

Python 侧 `rtp_llm/ops/rtp_llm/rtp_llm_op.py:26`：

```python
from rtp_llm.ops import RtpLLMOp as CppRtpLLMOp

class RtpLLMOp:
    def __init__(self, ...):
        self.ft_op = CppRtpLLMOp()              # ← 创建一个真正的 C++ 对象
    def start(self):
        self.ft_op.init(self.model, ...)        # ← 穿透胶水，调 C++ 函数
```

**关键理解**：
- `CppRtpLLMOp()` 创建的是**真正的 C++ 对象**，`self.ft_op` 只是"握着它的 Python 变量"
- `self.ft_op.init(...)` **执行的是 C++ 函数**，不是 Python 函数
- 销毁时 C++ 析构函数自动被调

🎯 pybind11 = **给 C++ 对象套了一层 Python 马甲**。外面看是 Python，里面干活的是 C++。

---

## 问题 3：C++ 也要反向读 Python 的数据（权重）

光有 Python→C++ 单向不够。模型权重是 PyTorch 张量，在 Python 侧。C++ 得能**反向**读。

📌 `RtpLLMOp.cc:106-137`：

```cpp
void RtpLLMOp::init(py::object model, py::object engine_config, ...) {
    //     ↑ py::object = "C++ 手里握着的 Python 对象引用"
    
    auto model_config = model.attr("model_config").cast<ModelConfig>();
    //                        ↑ 就像 Python 写 model.model_config
    
    py::object py_layers_weights = model.attr("weight").attr("weights");
    //                     一路 .weight.weights，就像 Python 那样
}
```

C++ 里能写 `.attr("xxx")`——pybind11 帮你做 Python 属性查找和类型转换。

🎯 pybind11 让 C++ 和 Python **互相能看见对方的对象**。

---

## 问题 4：gRPC server 在 C++ 里怎么跑起来？

📌 `RtpLLMOp.cc:127-133`：

```cpp
pybind11::gil_scoped_release release;
grpc_server_thread_ = std::thread(&RtpLLMOp::initRPCServer, this, ...);
grpc_server_thread_.detach();
```

**人话**：开一个后台线程专门接 gRPC 请求，主线程（Python）继续往下走。

后台线程里做的事（`RtpLLMOp.cc:318-333`）：

```cpp
grpc::ServerBuilder builder;
builder.AddListeningPort("0.0.0.0:<port>", grpc::InsecureServerCredentials());
builder.RegisterService(model_rpc_service_.get());    // 注册"接电话的服务"
grpc_server_ = builder.BuildAndStart();
grpc_server_->Wait();                                  // 永远阻塞
```

四步骤：创建 builder → 监听端口 → 注册服务 → 启动并永久阻塞。

---

## 问题 5：请求进来后，哪个 C++ 方法接住？

gRPC 规矩：**接口先用 `.proto` 声明，再由服务端实现**。

📌 `rtp_llm/cpp/model_rpc/proto/model_rpc_service.proto`（大致长相）：

```proto
service ModelRpcService {
    rpc GenerateStream(GenerateInputPB) returns (stream GenerateOutputsPB);
    rpc CheckHealth(EmptyPB) returns (CheckHealthResponsePB);
    ...
}
```

注意 `returns (stream ...)` 里的 **`stream`**——这就是 gRPC 的流式响应。

服务端实现 `LocalRpcServer.h:42-44`：

```cpp
grpc::Status GenerateStreamCall(
    grpc::ServerContext* context,
    const GenerateInputPB* request,
    grpc::ServerWriter<GenerateOutputsPB>* writer);
```

参数意义：
- `request`：一次性拿到所有输入（token_ids、config）
- `writer`：**一个可以反复写的管道**，每生成一个 token 就 `writer->Write(response)` 一次

---

## 问题 6：前端的 SSE 流，在 gRPC 层怎么实现？

核心答案：**`grpc::ServerWriter<T>`** 的 `Write()` 方法。

完整链条：

```
C++ 引擎生成 token
     │
     ▼
writer->Write(GenerateOutputsPB)           ← C++ 层
     │
     ▼ gRPC 底层封装成 HTTP/2 DATA frame
     │
     ▼ 网络传输
     │
     ▼ 前端 Python gRPC 客户端
     │
     ▼ async for token in stream:          ← 前端每次 yield 一个
     │
     ▼ FastAPI 把它包成 SSE: "data: xxx\n\n"
     │
     ▼ 浏览器收到
```

**协议栈对比**：
- gRPC 流 = HTTP/2 多帧
- FastAPI SSE = HTTP/1.1 chunked response
- 两者都是"不关连接反复发"，但协议不同
- **前端做了个转换**：把 gRPC 流接收到的每条消息，重新打包成 SSE 格式

---

## 问题 7：引擎启动时做了啥？

`RtpLLMOp::init()` 的完整流程：

```cpp
void RtpLLMOp::init(py::object model, py::object engine_config, ...) {
    // 1. 把 Python 传来的 config 转成 C++ 结构
    EngineInitParams params = initModel(model, engine_config, vit_config);
    
    // 2. 如果有投机解码模型，也转一遍
    auto propose_params = initProposeModel(propose_model, params);
    
    // 3. ⚠️ 放开 GIL
    pybind11::gil_scoped_release release;
    
    // 4. 开后台线程跑 gRPC server
    grpc_server_thread_ = std::thread(&RtpLLMOp::initRPCServer, ...);
    grpc_server_thread_.detach();
    
    // 5. 等 server 准备好才返回
    while (!is_server_ready_) { sleep(1); }
}
```

---

## 问题 8：为什么要 `gil_scoped_release`？—— GIL 死锁详解

Python 有个**全局解释器锁（GIL）**，同一时刻只允许一个线程执行 Python 字节码。

**场景**：Python 调进 C++ 的 `RtpLLMOp::init()`，按默认规则 C++ 里**还攥着 GIL**（防止随便回调 Python 时出事）。

**如果忘了 `gil_scoped_release`**，会出现经典死锁：

1. 主线程进 `init()` 后一直攥着 GIL
2. 主线程走到 `while (!is_server_ready_) sleep(1)` ——**睡觉时还攥着 GIL**
3. 后台线程执行 `initRPCServer`，一开头就要 `pybind11::gil_scoped_acquire`——要拿 GIL
4. 后台线程永远拿不到 GIL（主线程不放）
5. `is_server_ready_` 永远设不成 `true`
6. 主线程永远醒不了——整个进程卡死

**教训**：
- **任何 C++ 代码里要跑"不回调 Python 的长任务"之前，必须 `gil_scoped_release`**
- 什么时候要 `gil_scoped_acquire`？**要读 Python 对象 / 调 Python 函数时**。看 `RtpLLMOp.cc:288-289` 就是这么用的

---

## 问题 9：后端的优雅关闭

📌 `RtpLLMOp.cc:352-382`：

```cpp
void RtpLLMOp::stop() {
    if (grpc_server_) {
        // 等现有请求跑完
        while (auto onflight = model_rpc_service_->onflightRequestNum()) {
            sleep(1);
            if (超时) break;                 // 兜底
        }
        grpc_server_->Shutdown();
    }
}
```

和前端 `active_requests` 是同一套思想：不接新请求 → 等存量 → 超时强关。

---

## 🧠 合起来看：一次请求的"过桥之旅"

```
┌──────────────────────────────────────────────────┐
│ 后端进程（Python + 内嵌 C++）                     │
│                                                  │
│  【Python 层】                                   │
│  rpc_engine.py                                   │
│  └─ self.ft_op = CppRtpLLMOp()      pybind 创建 │
│  └─ self.ft_op.init(model, config, ...)         │
│            │                                    │
│            │ pybind11 过桥                      │
│            ▼                                    │
│  【C++ 层】                                      │
│  RtpLLMOp::init()                               │
│  ├─ 读 Python model 对象拿权重                   │
│  ├─ 构造 EngineInitParams                        │
│  ├─ gil_scoped_release                          │
│  └─ 开后台线程:                                  │
│        │                                        │
│        ▼                                        │
│     initRPCServer()                             │
│     ├─ grpc::ServerBuilder                      │
│     ├─ 注册 LocalRpcServiceImpl                  │
│     └─ server_->Wait()                          │
│                                                  │
│  【接到前端请求时】                               │
│  LocalRpcServer::GenerateStreamCall(req, writer)│
│  ├─ 塞进 EngineBase (下关主角)                   │
│  └─ 每生成一个 token: writer->Write(resp)       │
│            │                                    │
└────────────┼────────────────────────────────────┘
             │ gRPC 流
             ▼
        【前端进程】
```

---

## 📋 加一个新 gRPC 接口要改什么？

一份心智清单（以 `GetGpuMemoryUsage` 为例）：

| 改哪里 | 干嘛 |
|---|---|
| `model_rpc_service.proto` | 声明 rpc 和 message |
| Bazel 重新构建 | 自动生成 `.pb.h` / `.grpc.pb.h` |
| `LocalRpcServer.h` / `.cc` | 加虚方法实现 |
| `LocalRpcServiceImpl.h` | 转发到 LocalRpcServer |
| `grpc_client_wrapper.py` | 前端调用侧 |
| `frontend_app.py`（可选） | 暴露成 HTTP 路由 |

🎯 **记住**：**一个 gRPC 接口 = 一份 proto 声明 + 一段 C++ 实现 + 一份 Python 调用**。

---

## 📍 代码位置速查表

| 功能 | 位置 |
|---|---|
| Python 侧 pybind 包装 | `rtp_llm/ops/rtp_llm/rtp_llm_op.py` |
| C++ RtpLLMOp 类声明 | `rtp_llm/cpp/pybind/multi_gpu_gpt/RtpLLMOp.h` |
| C++ RtpLLMOp 实现 | `rtp_llm/cpp/pybind/multi_gpu_gpt/RtpLLMOp.cc` |
| pybind 注册入口 | `RtpLLMOp.cc:398` `registerRtpLLMOp` |
| 引擎 init | `RtpLLMOp.cc:106-137` |
| 模型参数收集 | `RtpLLMOp.cc:139-224` `initModel` |
| gRPC server 启动 | `RtpLLMOp.cc:283-335` `initRPCServer` |
| gRPC 服务实现 | `rtp_llm/cpp/model_rpc/LocalRpcServer.h/cc` |
| gRPC 接口声明 | `rtp_llm/cpp/model_rpc/proto/model_rpc_service.proto` |
| 流式响应核心 | `LocalRpcServer.h:42` `GenerateStreamCall` |

---

## 🎯 Level 2 关键记忆点

1. **后端 = Python 壳 + 内嵌 C++ 心脏**
2. **pybind11 让 Python 和 C++ 互相看见对方的对象**
3. **`py::object` = C++ 握着的 Python 引用，`.attr()` 读属性**
4. **GIL 是 Python 的锁，C++ 里长任务前必须 `gil_scoped_release`**
5. **gRPC 流式 = `ServerWriter<T>::Write()` 反复写**
6. **加新 gRPC 接口 = proto + C++ 实现 + Python 调用**
7. **优雅关闭和前端一个套路：等存量 → 超时强关**
