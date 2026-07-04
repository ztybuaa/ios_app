# 第二阶段训练与 iOS 端侧落地路线

## 当前基线

已完成一个可端侧复现的小模型基线：

- intent：字符 1-3 gram 线性分类器。
- slots：两个字符级 BIO span 标注器，分别抽取 `share_content` 与 `share_target`。
- 导出：`models/tiny_intent_slot_model.json`。
- 训练脚本：`scripts/train_tiny_intent_slot_model.py`。
- 评估报告：`reports/phase2_model_report.md`。

所有训练和评估命令必须使用：

```powershell
.\.venv\Scripts\python.exe
```

阶段是自检节点，不是一次对话只能推进一个阶段。只要验证闭环清楚，可以在同一次工作中连续完成数据整理、模型训练、Swift 推理器、App UI、资源模块和真机验证。

如需补充外部数据或下载预训练模型，必须放在工程内分层归档：

- `external_data/raw/<source_name>/`：原始下载数据。
- `external_data/labeled/<task_name>/`：人工或脚本标注后的数据。
- `external_data/processed/<task_name>/`：训练可直接消费的数据。
- `external_models/pretrained/<model_name>/`：外部预训练模型原件。
- `external_models/converted_ios/<model_name>/`：iOS 可部署转换产物。

每个外部来源都要配套说明来源、许可证、下载命令、字段含义、转换脚本和是否纳入训练/验证/测试。所有下载、转换、标注、训练、评估命令都必须在本工程 `.venv` 或对应工程环境中执行，不能使用 base 环境。

## 当前指标摘要

最终一次训练配置：

- 训练数据：`positive_train` + `with_unknown_train`，共 8800 条。
- intent epochs：10。
- slot epochs：12。
- 模型体积：约 1.09 MB。
- Python 开发机加载峰值内存：约 13.89 MB。
- Python 开发机平均推理耗时：约 0.76-1.02 ms/条。

关键评估：

- `positive_valid` 完整输出正确率：0.986。
- `positive_cold_start_test` 完整输出正确率：0.965。
- `with_unknown_valid` 完整输出正确率：0.940。
- `with_unknown_cold_start_test` 完整输出正确率：0.916。

当前短板：

- 带 unknown 冷启动集中失败在否定、延期、移动/重命名等非分享任务，以及部分长资源短语。
- `share_content` 在带 unknown 冷启动上低于目标槽位，需要下一轮加强 unknown 边界与资源短语边界。
- Python 延迟和内存只是开发机基线，最终必须在 iPhone App 内测量。

## 下一轮模型改进

1. 保持当前小模型作为可部署基线。
2. 增加 unknown hard negatives，不把“怎么操作”“先别”“明天再”“移动到”“重命名”等误触发为资源流程。
3. 针对长资源短语补充边界特征，尤其是：
   - `相册里的最新照片`
   - `下载的文件`
   - `下载的文件夹`
   - `孙会计的联系方式`
4. 增加按 intent 的错误分析表，不只看总分。
5. 保留 JSON 权重导出，避免引入无法在 iOS 复现的服务器依赖。

## iOS 端侧推理实现

Swift 侧建议先实现和 Python 完全一致的轻量推理器：

- `LinearClassifier.swift`
  - 读取 JSON 权重。
  - 根据 feature map 计算每个 label 分数。
  - 输出最高分 label。
- `TinyIntentSlotModel.swift`
  - 实现 intent features。
  - 实现 span tag features。
  - 分别运行 content span 和 target span 标注器。
  - intent 为 `unknown` 时强制返回空 slots。
- `SlotNormalizer.swift`
  - 复现 `scripts/prepare_dataset.py` 的 `normalized_slots` 逻辑。
  - 输出 `resource_type`、`resource_phrase`、`search_keyword`、`target`、`qualifiers`。

App 首屏需要展示：

- 输入原句。
- intent。
- 原始 slots。
- 标准化检索 slots。
- 推理耗时。
- 模型加载耗时。
- 当前进入的资源模块。

## 资源模块顺序

先做可单独测试的模块，不一次写完全部：

1. unknown 拦截：不进入任何资源流程。
2. contact：用 `target` 和 contact 类型 `search_keyword` 检索联系人候选。
3. photo：用 `search_keyword`、时间、选择提示检索相册候选。
4. file：用文件名、格式、索引摘要检索候选文件。
5. folder：用文件夹名、路径、子文件摘要检索候选文件夹。
6. video：用文件名、时间、封面/元信息检索候选视频。

每个模块都要单独记录：

- 输入槽位。
- 候选排序逻辑。
- 平均耗时。
- 峰值内存。
- 当前限制。

## Windows 与真机安装约束

当前 Windows 机器可以完成：

- 数据准备。
- 模型训练与评估。
- JSON 模型导出。
- Swift 源码和工程文件准备。

最终安装到 iPhone 真机仍需要 Xcode 相关真机运行链路，参考 Apple 的 Xcode 真机运行文档：

- https://developer.apple.com/documentation/xcode/running-your-app-on-simulated-or-physical-devices

如果本机没有 macOS/Xcode，需要接入已有 Mac、远程 Mac、云 Mac 或 CI 签名打包链路。签名已经准备好时，下一步要明确：

- Bundle ID。
- Team ID。
- Provisioning profile。
- 目标 iPhone 的 iOS 版本。
- 用本地 Mac 还是远程 Mac 执行 Xcode build/install。

## 下一步交付物

1. Swift 端 `TinyIntentSlotModel` 推理器。
2. 一个最小 SwiftUI Demo App：
   - 输入框。
   - 模型输出面板。
   - Debug 指标面板。
   - unknown 拦截。
3. 先接 contact 模块，再接 photo 模块。
4. 真机侧记录首载、单次推理、连续推理、内存和包体大小。
