# Changelog

## v0.1.0 (2026-09-27)

初始发布 ExplainForge —— 模型无关特征归因（XAI）系统。

- 复用 SOTA 方法论：KernelSHAP (Lundberg & Lee 2017) / LIME (Ribeiro 2016) /
  Permutation Importance (Breiman 2001)，并以有限差分 gradient*input 作经典强基线。
- 纯 numpy 实现 KernelSHAP + LIME 作为离线默认后端（零下载可跑）；`shap` 仅作可选参照校验。
- 统一 faithfulness 口径：分布保持的扰动式移除（列打乱）+ 全局特征排序 + Comprehension-AUC。
- 跨 3 个合成数据集取均值，给出 kernel_shap / lime / permutation / gradient_input 对照，
  含 naive 随机排序对照、oracle 上界、消融（联盟采样数）、失败案例分析、shap 排序一致性校验。
- 确定性：固定 seed 两次运行核心指标逐位一致；21 单测全绿。
- 作者：晨星。
