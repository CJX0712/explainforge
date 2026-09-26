# ExplainForge 架构文档

> 模型无关特征归因（XAI）系统。复用 KernelSHAP / LIME / 置换重要性 SOTA 方法论，
> 纯 numpy 离线兜底，确定性可复现，faithfulness 基准诚实可溯源。作者：晨星。

## 1. 设计原则

1. **模型无关**：所有解释器只依赖 `predict` / `predict_proba`，可套在任意黑盒分类器上。
2. **契约先行、单向无环**：`cli → pipeline → {data, explain, eval} → core`，无反向依赖。
3. **离线优先**：4 个核心解释器仅依赖 numpy / scikit-learn；`shap` 是可选参照校验，缺失不影响系统。
4. **确定性**：唯一 seed 入口 `core.seed.set_all(seed)`，所有采样（KernelSHAP 联盟、LIME 扰动、faithfulness 列打乱）受全局 RNG 控制，两次运行逐位一致。

## 2. 模块职责

| 模块 | 职责 | 关键产物 |
|------|------|----------|
| `core/seed` | 全局确定性 RNG 装配 | `set_all` / `rng` |
| `core/types` | `Attributions` / `Dataset` / `BenchmarkRow` 数据类 | 跨模块传递对象 |
| `core/errors` | E100–E106 错误码体系 | 可机器分支的错误 |
| `core/config` | `Config` + `EXPLAINFORGE_*` 环境变量覆盖 + 校验 | 单一可调参数源 |
| `core/interfaces` | `ModelProtocol` / `ExplainerProtocol` | 模块间契约 |
| `data/synthetic` | 含相关信号+冗余+噪声结构的数据生成 | `Dataset`（含 `signal_mask`） |
| `data/loaders` | CSV 载入（pandas / 纯 stdlib 兜底） | `Dataset` |
| `explain/kernel_shap` | 纯 numpy KernelSHAP（Shapley 核加权最小二乘） | `Attributions` |
| `explain/lime` | 纯 numpy LIME（加权岭归） | `Attributions` |
| `explain/permutation` | scikit-learn 置换重要性 | `Attributions`（全局） |
| `explain/gradient_input` | 有限差分 gradient×input（经典强基线） | `Attributions` |
| `explain/shap_backend` | 可选 `shap` KernelExplainer 参照 | 排序一致性校验 |
| `eval/faithfulness` | Comprehension-AUC / Sufficiency / Insufficiency | 可信度指标 |
| `pipeline/explain_pipeline` | 编排 + 跨数据集聚合 + 消融 + 失败分析 | `report` dict → `benchmark.json` |
| `cli` | argparse 入口，打印表 + 落盘 JSON | — |
| `examples/run_demo` | 端到端演示 | `benchmark.json` |

## 3. 调用图（acyclic）

```
cli.main()
   └─> ExplainPipeline.run()
          ├─> data.make_datasets()                 # 造数
          ├─> _make_model() + fit                  # 训黑盒
          ├─> _run_dataset()
          │     ├─> explain.build(name).fit().explain()   # 4 解释器 + naive + oracle
          │     ├─> eval.faithfulness_metrics()            # 可信度
          │     ├─> _ablation()                            # 联盟采样数消融
          │     ├─> _diagnostics()                        # 信用分散 / 低置信行为
          │     └─> _shap_validate()                       # 可选 shap 排序校验
          └─> 聚合 -> verdict(S/A/B/C) -> report
```

## 4. 接口清单（统一语义）

- 归因语义：**|值| 越大越重要**，跨方法一致，保证公平评测。
- 黑盒契约：`predict(X) -> labels`、`predict_proba(X) -> (N, C)`。
- 解释器契约：`fit(model, background) -> self`、`explain(X, model, class_index) -> Attributions | list[Attributions]`。
- 指标语义：Comprehension ∈ [0,1]，**越大越可信**；Sufficiency 越大越充分；Insufficiency 越大越 faithful。

## 5. 确定性保证

- `set_all(seed)` 一次性设齐 `random` / `numpy` 全局 / `np.random.Generator`。
- KernelSHAP 联盟掩码、LIME 扰动掩码均源自 `np.random.default_rng(seed)`，不依赖全局序列顺序。
- faithfulness 的列打乱使用固定 `_PERM_SEED=1234`，与业务 seed 解耦，保证指标稳定。
- demo 两次运行 `benchmark.json` 中 `kernel_shap.comprehension_auc` 逐位一致（已验证 `< 1e-9`）。

## 6. 离线兜底

- 重型/需网络的 `shap` 仅在 `_shap_validate` 中使用，且被 `try/except` + `available()` 探测包裹；
  不可用时 `shap_validation.skipped=true`，不伪造数字、不阻断流程。
- 4 个核心解释器与全部评测、单测均可在纯 numpy + scikit-learn 环境下零下载跑通。

## 7. 性能预算

- 默认 demo（RF 黑盒，nsamples=1024，3 数据集 × 60 测试样本）端到端约 14s（CPU），
  内存峰值 < 500MB，远低于 60s / 2GB 预算。
