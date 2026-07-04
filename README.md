# 端侧资源检索 Demo 数据与训练准备

本项目从六份 intent/slot 数据开始，目标是支撑后续 iPhone 端侧 Demo App：
小模型理解自然语言，抽取资源检索槽位，再进入 photo/video/file/folder/contact 候选检索流程。

## 重要运行规则

所有 Python 命令必须在本项目虚拟环境 `.venv` 下运行，不能使用 Anaconda base 环境。
项目脚本已加入环境守门逻辑：如果不是从本工程 `.venv` 启动，会直接退出。

首次准备环境：

```powershell
py -3 -m venv .venv
```

运行脚本时显式使用虚拟环境解释器：

```powershell
.\.venv\Scripts\python.exe .\scripts\prepare_dataset.py --root . --out .\processed --report .\reports\phase1_data_report.md
```

当前数据整理脚本只使用 Python 标准库，不需要安装额外依赖。

## 阶段推进规则

阶段是生成过程中的自检节点，不是对话边界。一次对话可以连续推进数据整理、训练、App 工程、资源模块和真机验证等多个阶段；每完成一个阶段要运行对应验证，并把结果归档到 `reports/`、`docs/` 或对应模块目录。

如果后续训练没有足够合适的数据，可以补充外部数据或训练好的模型，但必须遵守：

- 数据集放在工程内，按来源、任务和用途分目录归档。
- 不把原始数据、清洗数据、标注数据、训练输出揉在同一个目录。
- 每个外部数据源都要记录来源、下载时间、许可证或使用限制、转换脚本和字段说明。
- 外部预训练模型要放入独立模型目录，记录来源、版本、体积、转换方式和 iOS 可部署性。
- 所有下载、转换、标注、训练、评估命令都必须使用本工程 `.venv` 或对应工程环境，不能使用 base 环境。

推荐目录：

```text
external_data/
  raw/<source_name>/
  labeled/<task_name>/
  processed/<task_name>/
external_models/
  pretrained/<model_name>/
  converted_ios/<model_name>/
reports/
docs/
```

## 原始数据

五类正向数据：

- `train_intent_slots(1).json`
- `valid_intent_slots(1).json`
- `cold_start_test_intent_slots(1).json`

五类 + unknown 数据：

- `train_intent_slots_with_unknown.json`
- `valid_intent_slots_with_unknown(1).json`
- `cold_start_test_intent_slots_with_unknown.json`

## 第一阶段输出

- `processed/dataset_manifest.json`：六份数据的 manifest、schema 和统计。
- `processed/datasets/*.jsonl`：每份原始数据对应一份标准化 jsonl。
- `processed/datasets/all.jsonl`：六份数据合并后的标准化 jsonl。
- `reports/phase1_data_report.md`：第一阶段数据整理报告。

## 标准化槽位

原始监督信号保留为：

- `intent`
- `raw_slots.share_content`
- `raw_slots.share_target`

额外派生给资源检索模块使用的字段：

- `normalized_slots.resource_type`
- `normalized_slots.resource_phrase`
- `normalized_slots.search_keyword`
- `normalized_slots.target`
- `normalized_slots.target_keyword`
- `normalized_slots.qualifiers`

对 `这张照片`、`这个文件` 这类泛指资源，`search_keyword` 显式为 `null`，不虚构关键词；后续由当前选择、最近资源或资源模块上下文处理。

## 当前验证结果

已用 `.venv\Scripts\python.exe` 运行第一阶段脚本：

- 共生成 13200 条标准化记录。
- 六份原始 JSON 均可解析。
- `unknown` 样本的标准化槽位为空，不会触发资源检索流程。

## 第二阶段小模型基线

训练命令：

```powershell
.\.venv\Scripts\python.exe .\scripts\train_tiny_intent_slot_model.py --data-dir .\processed\datasets --model-out .\models\tiny_intent_slot_model.json --metrics-out .\reports\phase2_model_metrics.json --report-out .\reports\phase2_model_report.md --tag-epochs 12
```

当前输出：

- `models/tiny_intent_slot_model.json`
- `reports/phase2_model_metrics.json`
- `reports/phase2_model_report.md`
- `docs/phase2_model_and_ios_plan.md`

当前基线是字符级轻量模型，导出为 JSON 权重，后续 Swift 端可以复现同样特征和解码逻辑。

## iOS Demo 工程

完整 iOS Demo 已生成在：

- `ios_app/IntentResourceDemo.xcodeproj`
- `ios_app/IntentResourceDemo/`

说明文档：

- `docs/ios/README_ios_demo.md`

本地结构验证：

```powershell
.\.venv\Scripts\python.exe .\scripts\validate_ios_project.py
```

## GitHub Actions 打包 IPA

已加入 GitHub Actions 工作流：

- `.github/workflows/build-ios-unsigned-ipa.yml`
- `scripts/ci/build_unsigned_ipa.sh`

用途：在 GitHub 的 macOS runner 上编译工程并产出未签名 IPA：

- `IntentResourceDemo-unsigned.ipa`

然后你可以在 iPhone 的签名工具中用 P12 和描述文件给这个 IPA 签名安装。

详细步骤：

- `docs/github/build_ipa_with_github_actions.md`
