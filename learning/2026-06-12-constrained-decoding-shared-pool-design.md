# 受限解码 · 共享底池设计(草稿)

> 状态:设计讨论记录(草稿,未定稿)。日期:2026-06-12。
> 目的:把"动态/共享底池下的受限 beam search"升级方案记录下来,供后续细化为正式 spec。

---

## 1. 背景与问题

生成式检索(LLM 逐 token 生成商品/主播的 Semantic ID, 简称 SID)需要**受限 beam search**,保证生成结果落在"可分发底池"内,去掉非法结果、提升生成 quota 利用率。

底层能力已具备:CSR 受限解码(代码 "ele 实时限制编码")。

- 编译:`csr_utils.h: build_csr_from_fresh_data`(SID 列表 → CSR 转移矩阵)
- 编排:`TreeLogitsProcessorCSR`(.h / CSRGpu.cc)
- GPU kernel:`csr_logits.cu`(mask 生成 / 状态更新 / gather)
- 入参:`generate_config` 的 `ele_rq_ids` / `ele_rq_ids_pb` / `extra_info`

**当前机制 = 方案 A(逐请求,无共享)**:

- `GenerateStream.cc:89` 在 per-request stream 构造时调 `createLogitsProcessors`
  → 从该请求的 `ele_rq_ids` 后台 `build_csr`(与 prefill 重叠)
  → 挂在 per-request `logits_processor_list_`(GenerateStream.h:639)
  → 请求结束即丢。
- 全仓库**无任何跨请求 CSR/底池缓存**。

**痛点**:当底池是"共享、慢变"的(直播间),同一份池被成千请求各自重传 + 重建 + 重 H2D,开销是 `O(QPS × 池大小)`,而本应是 `O(池版本数 × 池大小)`。

---

## 2. 两个目标场景

| 场景 | 底池特征 | 规模 |
|---|---|---|
| 直播间(主搜实时直播间) | 共享、分钟级慢变(在播主播) | 瞬时在播约 2w–4w(35w 为 6 个月宇宙/内存上限) |
| 电商店铺(原 ele 场景) | per-query 唯一(命中 ∩ 有货 ∩ 地理) | 每 query 不同 |

设计目标:**一套机制优雅吃下两者(以及任意受限编码场景),能合进主干。**

---

## 3. 核心设计:内容寻址的"已编译约束索引"缓存

把现有 `TreeLogitsProcessorCSR` 拆成四层(关键重构):

```
1. ConstraintSpec     —— "允许什么"的可序列化描述(今天 = SID 列表;未来可扩 grammar/regex)
2. ConstraintCompiler —— Spec → 不可变、常驻 GPU 的 ConstraintIndex(今天 = build_csr_from_fresh_data)
3. ConstraintIndexCache —— 内容寻址(按 hash(Spec) 缓存)+ LRU/TTL + 引用计数 + RCU 原子切换
4. ConstraintRuntime  —— 每请求私有的解码状态机:mask(logits, state) + advance(state, token)
```

**最关键的重构**:把"不可变的已编译索引(可共享)"与"每请求的解码状态(私有)"**拆开**。
现有 `TreeLogitsProcessorCSR` 把两者揉在一起(又建树又持状态),所以无法共享。拆开后:共享成为可能,且代码更干净——**此拆分与做不做共享池无关,本身就是更好的设计。**

**安全基石:缓存是纯优化,永不为真相。**
命中即加速;未命中就重建(或让上游补发数据)。correctness 永不依赖缓存 →
无缓存失效协议、无一致性 bug、LRU/TTL 可随意驱逐、进程重启冷缓存按需重建。

---

## 4. 统一的两个入口(同一套实现)

请求用**两种方式之一**指定底池,它们只是"取缓存钥匙(指纹)"的两种方式:

```python
def resolve(request) -> ConstraintIndex:
    if request.pool_id:                          # 具名(直播间):名字 → 指纹
        key = registry[request.pool_id]
    else:                                        # 内联(电商):值 → 指纹
        key = content_hash(request.pool_inline)
    return cache.get_or_build(key, ...)          # ← 往后全部合流,无分叉

def update_pool(name, sids):                     # 推送(仅具名需要),也复用同一缓存
    key = content_hash(sids)
    cache.get_or_build(key, sids)                # 与内联走同一条编译+缓存
    registry[name] = key                         # 仅额外记一笔"名字 → 指纹"别名
```

**唯一的代码分叉就是 `resolve()` 里的 if-else(2 行);`cache.get_or_build` 之后(编译、缓存、显存、运行时)完全共用。** 不是两套实现,是一套引擎开两个薄前门。

唯一新增物件:`map<名字, 指纹>` 注册表 + `update_pool` 接口(几十行,且可选——纯内联场景不需要它)。

---

## 5. 两个场景怎么用

**直播间(共享池)**

1. 算法控制面每分钟(或 saro 变更时)算出当前在播 SID → 调 `update_pool("live", sids)`。
2. 引擎编译 CSR、RCU 切到新版本(常驻显存)。
3. 每个查询:推理请求只带 `pool_id="live"`(默认 latest)。**零池传输、零重建、零 H2D。**

**电商(per-query 唯一池)**

1. 每个 query,检索上游算出候选店铺。
2. 推理请求内联带 `pool_inline=[...]`。
3. 引擎 hash → 几乎必然 miss → 当场建一次(后台与 prefill 重叠,同今天)→ 用完丢。**行为与今天一致,零回退。**
4. 白捡:若两个 query 候选池碰巧相同(同城同类目),第二个自动命中缓存,无需声明。

**生命周期差异(非代码分叉,仅引用归属不同)**:
- 具名:registry 钉住引用 → index 长命、跨请求共享。
- 内联:仅 LRU 缓存 + 当前请求持引用 → 用完即弃。

---

## 6. 多副本

线上引擎多副本,每副本各自持有 GPU 常驻 index。

- **地板成本(躲不掉)**:每版本 × 每副本 推送一次(2w–4w ≈ 几百 KB/min/副本) + 编译一次(毫秒级)。极小。
- **push 扇出方式(待定)**:
  - (a) 直接推:控制面知道副本列表,挨个 push。简单,但控制面管副本成员。
  - (b) 推到小"池服务"/共享存储,副本订阅拉取。解耦、好扩。

---

## 7. 明确不做(YAGNI)

- **增量推送**(只发上下播 delta):省传输但 CSR 仍需整体重建;全量推才几百 KB/min,不值得上 delta 协议。
- **离线编译 + 分发 CSR 字节**:编译本就毫秒级,还耦合显存布局/vocab,不值。

> 仅当池规模或更新频率大幅上升时才重新评估。

---

## 8. 待定决策(下一步要拍的)

1. 多副本 push 扇出方式:直接推 vs 小池服务(取决于副本规模与现有基础设施)。
2. 缓存键 `content_hash` 算什么:是否需要把 vocab_size / base id 版本一并纳入(换模型/换 tokenizer 时缓存自然失效)。
3. `ConstraintSpec` 是否一步到位通用化(grammar/regex/FST)还是先只做 SID 列表(倾向先只做 SID,接口留扩展位)。
4. 适配新模型的两个硬编码点:`csr_utils.h:45` 的 `<shop_0_0/1_0/2_0>` token 名、`token_num=3` 层数——是否在本次一并改为可配置。
5. `pool_id` / `pool_inline` 字段如何加入 proto / GenerateConfig,以及 `update_pool` 走哪个 RPC 面。

---

## 9. 与论文 / 业界

- **STATIC 论文(arXiv 2602.22647)** 明确把转移矩阵 T 的构建当作**离线/周期性一次性成本**,其 YouTube 部署是中心化维护的共享底池——即论文默认形态就是"共享池",逐请求建反而是本仓库的简化。本设计是向论文原形回归。
- **vLLM / SGLang / XGrammar / Outlines** 的结构化输出:按语法/正则 hash **缓存已编译的约束(FSM/grammar)**,大量请求共享——与本设计同一范式。
- 通用工程:内容寻址(Bazel/Nix 构建缓存)、HTTP ETag / If-None-Match、Docker 层 have/want——同一思想。
