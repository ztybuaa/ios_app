# Chinese-CLIP RN50 FP16 迁移前评测

## 结论

本报告使用官方 RN50 PyTorch checkpoint 评估原生中文检索，并用相同提示词和裁剪规则校准待转换的 FP16 Core ML 模型。Core ML 数值一致性由 macOS 转换流程单独验证。

- checkpoint SHA-256: `b196ee3ee528b70be1158ab1aafb1d2f1c801ad2d9ffb3bae31b0d305f82fc88`
- 中文查询数: `6`
- 图片数: `10`
- minimum similarity: `0.470`
- 普通查询 minimum margin: `0.012`
- 普通截图 minimum margin: `0.011`

## 结果

| 模式 | Top-K | 标注目标覆盖 | 高精度门限 | 已知误检 | 结论 |
|---|---:|---:|---:|---:|---|
| app_full_quality | 6/6 | 7/9 | 6/6 | 0 | 通过 |
| six_view_stress | 6/6 | 9/9 | 6/6 | 0 | 通过 |

## 六视图压力测试明细

| 查询 | 通过候选 | Top-3 |
|---|---|---|
| 小猫图片 | cats (full, m=0.0351) | cats (full, m=0.0351), corgi (center, m=-0.0374), beach (top-right, m=-0.0433) |
| 小狗图片 | corgi (full, m=0.0429) | corgi (full, m=0.0429), beach (center, m=-0.0497), cats (top-left, m=-0.0506) |
| 风景图片 | moraine_lake (full, m=0.0602), beach (top-right, m=0.0168) | moraine_lake (full, m=0.0602), savanna (center, m=0.0256), beach (top-right, m=0.0168) |
| 美女图片 | portrait_woman (bottom-left, m=0.0741), woman_afro (bottom-left, m=0.0380) | portrait_woman (bottom-left, m=0.0741), woman_afro (bottom-left, m=0.0380), savanna (top-right, m=-0.0007) |
| 截图 | game_screenshot (full, m=0.0739), semantic_search_ui (full, m=0.0344) | game_screenshot (full, m=0.0739), semantic_search_ui (full, m=0.0344), beach (full, m=-0.0023) |
| 游戏截图 | game_screenshot (full, m=0.0263) | game_screenshot (full, m=0.0263), savanna (top-right, m=0.0063), beach (full, m=-0.0189) |

## 限制

当前集合只有 10 张图片和 6 条查询，足以拦截已知猫狗、风景、人物、截图和游戏截图回归，但不能替代真实相册评测。最终门限还需结合 FP16 Core ML 数值一致性和真机相册结果复核。
