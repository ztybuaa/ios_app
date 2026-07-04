# iOS 端侧 Demo 工程说明

工程位置：

- `ios_app/IntentResourceDemo.xcodeproj`
- App 源码：`ios_app/IntentResourceDemo/`

## 已实现内容

1. 小模型端侧推理
   - 模型文件：`ios_app/IntentResourceDemo/Resources/tiny_intent_slot_model.json`
   - Swift 推理：`NLP/TinyIntentSlotModel.swift`
   - intent：字符 n-gram 线性分类器
   - slots：`share_content` 和 `share_target` 两个 BIO span 抽取器

2. 槽位标准化
   - `NLP/SlotNormalizer.swift`
   - 输出 `resource_type`、`resource_phrase`、`search_keyword`、`target`、`qualifiers`

3. 五类资源模块
   - contact：`Contacts` 读取通讯录，按槽位搜索联系人候选
   - photo：`Photos` + `Vision`，按时间/截图/视觉标签/关键词别名筛选照片
   - video：`Photos` + 视频缩略帧 `Vision` 标签筛选视频
   - file：工程内 `sample_resource_index.json` 端侧索引检索
   - folder：工程内 `sample_resource_index.json` 端侧索引检索
   - unknown：直接拦截，不进入任何资源模块

4. Debug UI
   - 输入自然语言
   - 展示 intent、原始 slots、标准化 slots
   - 展示模型加载耗时、单次推理耗时、当前内存
   - 展示资源候选和目标联系人候选

## 真机运行

在 macOS + Xcode 上打开：

```text
ios_app/IntentResourceDemo.xcodeproj
```

需要在 Xcode 中设置：

- `PRODUCT_BUNDLE_IDENTIFIER`
- `DEVELOPMENT_TEAM`
- 你的 iPhone 设备

当前 project 已开启 Automatic Signing，但 `DEVELOPMENT_TEAM` 留空，需要按你的开发签名填写。

## 权限

`Info.plist` 已声明：

- `NSContactsUsageDescription`
- `NSPhotoLibraryUsageDescription`

权限被拒绝时，App 会在状态区域显示明确提示，不会静默失败。

## 文件与文件夹模块说明

iOS App 不能任意全盘扫描用户 Files App 中所有文件和文件夹。当前 Demo 使用工程内端侧索引：

- `ios_app/IntentResourceDemo/Resources/sample_resource_index.json`

后续可以扩展为：

- 用户通过 Document Picker 导入文件后建立本地索引
- 使用 App Group 或应用沙盒内文件建立索引
- 用 Core Spotlight 索引 App 自己拥有的数据

## 本地验证

所有 Python 验证必须使用本工程 `.venv`：

```powershell
.\.venv\Scripts\python.exe .\scripts\validate_ios_project.py
```

该验证会检查：

- Swift 文件是否齐全
- 模型 JSON 是否放入 App Bundle 资源目录
- 样例资源索引是否可解析
- Info.plist 权限声明是否存在
- Xcode project 是否引用关键源码和资源

## 阶段检查

本工程已经按阶段完成：

1. 数据整理：`reports/phase1_data_report.md`
2. 小模型训练：`reports/phase2_model_report.md`
3. iOS 推理移植：`ios_app/IntentResourceDemo/NLP/`
4. 资源模块：`ios_app/IntentResourceDemo/ResourceModules/`
5. UI 与权限：`ios_app/IntentResourceDemo/Views/`、`Support/Info.plist`
6. 工程归档与验证：`docs/ios/README_ios_demo.md`、`scripts/validate_ios_project.py`
