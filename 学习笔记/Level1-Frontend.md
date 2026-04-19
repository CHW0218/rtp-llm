# Level 1：Frontend — 从 5 行代码进化到完整前端服务

## 🎯 本关核心心智模型

Frontend 不是一坨复杂代码。它是**从 5 行 FastAPI 起步，每遇到一个问题就加一层**长出来的。读这段代码时，心里要问的不是"这段在干嘛"，而是"**这段在解决哪个问题**"。

Frontend 负责：
- 收 HTTP 请求
- 把文字变成 token（分词）
- 把 OpenAI 格式翻译成模型能吃的
- 把后端生成的 token 流式推送给用户

Frontend **不**负责：
- 跑模型（后端的事）
- 调度 / 批处理（Scheduler 的事）
- 任何 GPU 计算

一句话：**翻译 + 传话员**。

---

## 📍 起点：一个最最最简陋的版本

```python
def chat(prompt):
    return model(prompt)

print(chat("帮我写首诗"))
```

能跑！但只是命令行脚本。接下来每遇到一个问题，就加一层。

---

## 问题 1：怎么让别人访问我的模型？

**答**：开一个 HTTP 端口。Python 里最舒服的工具是 FastAPI。

```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/chat")
def chat(prompt: str):
    return model(prompt)
```

这就是 `frontend_app.py` 最本质的骨架。

📌 `frontend_app.py:170` `app = FastAPI(middleware=middleware)`

---

## 问题 2：模型其实听不懂人话

模型只认**数字**。需要加一步**分词（tokenize）**："帮我写首诗" → `[1234, 567, ...]`。

```python
token_ids = tokenizer.encode(prompt)      # 文字 → 数字
output_ids = model(token_ids)
output_text = tokenizer.decode(output_ids) # 数字 → 文字
```

📌 `frontend_server.py:471` `tokenizer_encode()`

### 🔍 深挖：token_id 到 embedding 还要转换吗？

需要，但**不是 Frontend 做的，是模型自己做的**。

```
"你好"
  ↓ tokenizer.encode()              ← Frontend 做
[1234, 5678]
  ↓ 通过 gRPC 发给后端
  ↓ 进入模型第一层：Embedding Layer  ← 模型自己做
[[0.12, -0.34, ...], [0.78, 0.55, ...]]  ← 向量
  ↓ Transformer 层 × N
  ↓ 输出
```

**Embedding 层本质是一张大表**（矩阵）：
- 行数 = 词表大小（如 Qwen 有 152000 行）
- 列数 = hidden_size（如 4096）
- `token_id=1234` → 去第 1234 行取出那 4096 个数字
- 这张表是**模型权重的一部分**，训练时学出来的

**一句话记住**：
- Frontend 做"字 → 数字"（tokenize）
- 模型做"数字 → 向量"（embedding lookup）
- 两件事，两个地方。

---

## 问题 3：生成太慢，用户盯着空白屏幕等

模型一个字一个字蹦出来——没必要攒完 500 字才给用户。

→ **流式推送（streaming）**。每生成一个 token 就推一个。

HTTP 里实现流式的协议叫 **SSE（Server-Sent Events）**：

```
data: {"token": "床"}

data: {"token": "前"}

data: [DONE]
```

📌 `frontend_server.py:180` `stream_response()`

### 🔍 深挖：SSE 到底是什么？和 Socket 有什么区别？

| | Socket | WebSocket | SSE |
|---|---|---|---|
| 是什么 | 最底层 TCP/UDP 连接 | HTTP 升级的双向通道 | 一个 HTTP 响应，但"一直不结束" |
| 方向 | 双向 | 双向 | 单向（服务器 → 客户端） |
| 协议层 | L4 传输层 | 自定义协议 | HTTP 之上的文本格式约定 |
| 穿透代理/CDN | 看情况 | 经常被挡 | 完美（就是 HTTP） |
| 自动重连 | 自己写 | 自己写 | 浏览器自动 |
| 复杂度 | 高 | 中 | 超低 |

**SSE 的本质**：普通 HTTP 是"一次性返回完就关"；SSE 把它改成"**写一点 → 客户端收一点 → 连接不关**"，直到服务器说"我讲完了"。

**数据格式**：每条消息 `data:` 开头，**两个换行 `\n\n`** 结尾。完毕。

**为什么 LLM 用 SSE 不用 WebSocket？**
1. 单向够用（用户发一次请求后就只管听）
2. SSE 就是 HTTP，防火墙/CDN/代理全部畅通无阻
3. 实现超简单：`yield f"data: {x}\n\n"` 就完事

📌 真实代码：
- `frontend_server.py:190` `yield response_data_prefix + data_str + "\r\n\r\n"`
- `frontend_server.py:444` `StreamingResponse(..., media_type="text/event-stream")`

---

## 问题 4：Python 写 HTTP 爽，但跑模型慢

矛盾点：
- HTTP 层用 Python 爽（FastAPI、异步 IO、库多）
- 模型推理必须用 C++/CUDA 才够快

→ **拆成两个进程**：
- 前端进程：纯 Python，只管 HTTP
- 后端进程：C++，只管跑模型
- 中间用 **gRPC** 打电话

gRPC 可以理解为"**两个进程之间打电话的高效协议**"，自带类型检查，比 HTTP 更快。

📌 `frontend_app.py:81` `self.grpc_client = GrpcClientWrapper(...)`

### 🔍 深挖：真的只是因为 C++ 快吗？为什么不用 pybind 同进程调？

**光为了性能不需要拆进程**——pybind11 就够了。rtp-llm 自己也大量用 pybind11。

真正拆进程的三个原因：

**① 部署拓扑灵活性（最重要）**

前后端的"形状"完全不一样：

| | Frontend | Backend |
|---|---|---|
| 瓶颈 | 网络 IO | GPU 算力 |
| 扩容单位 | 随意，10 个副本也行 | 每个 GPU 一个 |
| 加载时间 | 秒级 | 分钟级（加载权重慢） |

更极端的：这个项目支持 **PD 分离（Prefill/Decode 分离）**——处理 prompt 的机器和生成 token 的机器可以是**不同物理机**。没进程隔离根本做不到。

📌 `frontend_app.py:68` 的 `separated_frontend` 开关。

**② 故障隔离**
- 前端崩了 → 只丢 HTTP 层，GPU 进程没事，模型不用重新加载
- 后端崩了 → 前端还能返回"服务不可用"给用户

**③ Python GIL**

GIL 让 Python 没法真并行。HTTP 服务和模型推理共进程会互相抢 GIL。拆开后前端专心 IO，后端是薄薄一层壳调 C++，GIL 竞争就没了。

**一句话记住**：**拆进程不是因为 Python 慢，而是因为前后端"形状不一样"，放一起会互相掣肘。**

---

## 问题 5：两个进程启动顺序怎么办？

前端先起来，后端还在加载模型——用户请求打进来就崩。

→ **前端启动时反复 ping 后端**，等后端说"我好了"才开门营业。

📌 `frontend_app.py:90-116` `_wait_backend_health_ready_impl()`

---

## 问题 6：兼容 OpenAI API 格式

OpenAI 的格式是结构化消息：

```json
{"messages": [
  {"role": "system", "content": "你是助手"},
  {"role": "user", "content": "你好"}
]}
```

但模型只吃**一个长字符串**。中间差一步——**chat template**：

```
<|system|>你是助手<|user|>你好<|assistant|>
```

每个模型模板都不一样（Llama、Qwen、DeepSeek 各一套）。所以有专门的 `OpenaiEndpoint` 类。

📌 `frontend_server.py:115` `self._openai_endpoint = OpenaiEndpoint(...)`

---

## 问题 7：运维要重启，还有用户在生成中！

→ **优雅关闭（graceful shutdown）**：
1. 停止接收新请求
2. 等现有请求跑完
3. 才真正退出

用全局计数器 `active_requests` 跟踪"有多少请求在跑"，关闭时等它变 0。

**工程细节**：真实生产里运维不是直接 kill，而是：
1. 发 **SIGTERM**（"你准备关吧"）
2. 进程触发 `shutdown()`
3. 等超时（如 30 秒），还没关完才发 **SIGKILL** 强杀

📌 `frontend_app.py:41-53` `GracefulShutdownServer.shutdown()`

---

## 问题 8：有人疯狂刷接口怎么办？

→ **并发控制**：最多同时 N 个请求，超了拒绝（抛 `ConcurrencyException`）。

📌 `frontend_server.py:62` `self._global_controller`

---

## 🧠 Level 1 心智图

```
用户 HTTP 请求
     │
     ▼
 ┌──────────────────────┐
 │  FastAPI (Uvicorn)   │  ← 监听端口
 └──────────┬───────────┘
            │
 ┌──────────▼───────────┐
 │   FrontendApp        │  ← 路由分发 + 并发控制
 └──────────┬───────────┘
            │
 ┌──────────▼───────────┐
 │  FrontendServer      │  ← 业务逻辑、日志、指标
 └──────────┬───────────┘
            │
 ┌──────────▼───────────┐
 │  OpenaiEndpoint      │  ← 模板渲染 + 分词
 └──────────┬───────────┘
            │ gRPC
            ▼
      【后端进程】
```

---

## 📋 每一层为什么存在（总结表）

| 层 | 解决的问题 |
|---|---|
| FastAPI 路由 | 别人怎么访问我 |
| tokenizer | 模型只认数字不认字 |
| SSE 流式返回 | 用户不想盯着空白屏幕 |
| gRPC 调后端 | 前后端部署/故障/GIL 解耦 |
| 健康检查 | 前后端启动顺序 |
| OpenaiEndpoint | 兼容 OpenAI 客户端 |
| GracefulShutdown | 重启不丢请求 |
| 并发控制 | 防刷防爆显存 |

---

## 📍 代码位置速查表

| 功能 | 位置 |
|---|---|
| FastAPI app 创建 | `frontend_app.py:170` |
| OpenAI 聊天路由 | `frontend_app.py:319` |
| 通用推理路由 | `frontend_app.py:306` |
| 健康检查 | `frontend_app.py:195` |
| 等后端就绪 | `frontend_app.py:90-116` |
| 优雅关闭 | `frontend_app.py:41-53` |
| 业务处理主类 | `frontend_server.py:43` `FrontendServer` |
| 流式响应发射器 | `frontend_server.py:180` |
| 非流式处理 | `frontend_server.py:420` |
| tokenize 接口 | `frontend_server.py:471` |
| OpenAI 端点实例化 | `frontend_server.py:115` |
| 并发控制器 | `frontend_server.py:62` |

---

## 🎯 Level 1 关键记忆点

1. **Frontend 是层层堆出来的**，每一层对应一个具体问题
2. **tokenize 在 Frontend，embedding 在模型内部**（第一层）
3. **拆进程不是为了性能**，是为了部署灵活性 + 故障隔离 + GIL
4. **SSE 就是"一直不关的 HTTP 响应"**，格式是 `data: xxx\n\n`
5. **gRPC 是两个进程之间打电话**的协议
6. **优雅关闭 = 等存量请求跑完再退出**
