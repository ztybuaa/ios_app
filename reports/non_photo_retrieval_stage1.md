# 非照片资源检索 Stage 1 报告

## 可回退照片基线

- Git commit：`5235e6427b35c13e24a5f383b0175e595d61e633`
- annotated tag：`checkpoint/photo-search-v1.8-build9`
- 已验证 IPA SHA-256：`fd9864b31c85dd7da561bf4635ea599610ce3d8f5fb33670be45e2c07b9eafba`
- 后续实验分支：`codex/non-photo-retrieval`
- 实验 App 版本：`1.9 (10)`

照片基线的 Chinese-CLIP RN50 模型、相似度阈值、高质量重排和缓存策略均未在 checkpoint 上改写。

## 任务与约束

- 任务：自然语言分享指令先预测 `photo/video/file/folder/contact/unknown`，再在对应资源域内检索候选。
- 意图模型：现有字符 n-gram 线性分类器和双 BIO 槽位模型，不更换模型家族。
- 主评测：`with_unknown_cold_start_test`，六类各 200 条。
- 候选指标：Top-1、MRR、Recall@3、开放集误报率。
- 端侧约束：离线运行；不依赖系统翻译；iOS App 不能任意扫描其他 App 的 Files 沙盒。

## 根因审计

`with_unknown_cold_start_test` 的总体意图准确率为 `0.9825`。逐类 F1 为 photo `0.9676`、video `0.9824`、file `0.9779`、folder `0.9924`、contact `0.9926`、unknown `0.9823`。因此主要问题不是类型预筛选，而是类型确定后的候选召回和排序。

候选层原有四个共性问题：

1. 时间、格式和选择条件会污染语义关键词，例如把“最新的视频”归一化为 `最新`。
2. “合同、报告、截图、录屏”等真实主题被当作资源类别词删除。
3. 文件、文件夹和联系人把所有字段拼接后只做整串 `contains`，没有字段权重、中文近似匹配、电话规范化或拒绝阈值。
4. `recent` 只给所有候选同时加 `0.25`，未执行日期过滤，也不能改变有效排序。

## 离线候选实验

固定夹具包含 23 个候选、20 条闭集查询和 8 条开放集查询，覆盖文件、文件夹、联系人、中文字符二元组、拼音、电话、邮箱、组织加职位、格式和日期条件。它只用于复现已知故障模式，不代表真实用户数据达到 100%。

| 方法 | Top-1 | MRR | Recall@3 | 开放集误报 |
| --- | ---: | ---: | ---: | ---: |
| 原 CandidateScorer | 0.550 | 0.565 | 0.550 | 2/8（25%） |
| 字段感知排序 | 1.000 | 1.000 | 1.000 | 0/8（0%） |

机器可读结果：`reports/non_photo_retrieval_stage1_metrics.json`。

## 视频代理实验

现有照片评测不能证明视频时间轴质量。为了只验证 Stage 1 的“单海报帧”可行性，将 COCO128 的 128 张公开图片当作每段视频的一张代表帧，对 10 个类别比较查询写法：

| 提示词 | Top-1 | macro P@5 | 现有门控 precision |
| --- | ---: | ---: | ---: |
| `主体+图片` | 7/10 | 0.500 | 0.787879（26 TP / 7 FP） |
| `主体+视频` | 6/10 | 0.480 | 0.677419（21 TP / 10 FP） |
| 仅主体 | 7/10 | 0.460 | 0.529412（9 TP / 8 FP） |

因此 video-poster 路径把真实主题规范成图片域提示；泛指、日期、格式、方向和时长请求先走 PhotoKit 元数据，不把这些条件送入 RN50。240 个普通视频需要 304 次两阶段图像预测，500 个猫/人物视频需要 564 次，分别比六视图穷举少约 78.9% 和 81.2%。这些数字不包含 PhotoKit 冷缓存读取时间。

机器可读结果：`reports/video_poster_prompt_proxy_metrics.json`。脚本 `scripts/eval_video_poster_prompt_proxy.py` 已接入本地预检和 CI。

单海报帧仍会漏掉只在中后段出现的目标和纯语音内容。Stage 2 需要在真视频上评估 10%/50%/90% 多帧索引；音频语义另走转写，不应由图片模型猜测。

## Stage 1 实现

- Query normalization：结构条件与主题分离，同时保留合同、报告、截图、录屏等主题。
- Video：照片调用保持原路径；有真实主题的视频使用 RN50 海报帧两阶段检索，结构查询使用日期、格式、方向和时长硬过滤。
- File/folder：字段权重、中文字符二元组、格式/日期硬过滤、最近更新时间排序、文件正文词、目录子项聚合词和开放集拒绝。
- Contact：姓名、昵称、拼音、组织、部门、职位、电话、邮箱、地址、社交资料和 IM 分字段排序；缓存联系人并在 `CNContactStoreDidChange` 后失效。
- Error contract：Bundle 索引损坏不再静默变成空索引；资源失败和目标联系人失败分别报告；缺少当前选择上下文时不拿最近资源替代。

## 能力边界

| 类型 | Stage 1 可测能力 | 仍缺少 |
| --- | --- | --- |
| photo | checkpoint 中已验证的 RN50 两阶段检索 | 当前选择对象的上游上下文 |
| video | PhotoKit 元数据过滤和单海报帧语义检索 | 真视频阈值、多帧、音频语义 |
| file | Bundle 样例索引上的字段排序 | Document Picker 授权、真实文件索引、正文抽取 |
| folder | Bundle 样例索引上的名称/摘要/标签排序 | 授权目录层级、子文件命中聚合 |
| contact | 系统 Contacts 的结构字段和拼音排序 | 创建时间、最近通话、短信、微信二维码等系统不可访问数据 |

真实 Files 方案必须先由用户通过 [UIDocumentPickerViewController](https://developer.apple.com/documentation/uikit/uidocumentpickerviewcontroller) 授权文件或目录，再对 App 已获授权的内容建立索引；[CSSearchableIndex](https://developer.apple.com/documentation/corespotlight/cssearchableindex) 只索引本 App 提交的内容，不能绕过系统沙盒。视频多帧阶段可使用 [requestAVAsset](https://developer.apple.com/documentation/photos/phimagemanager/requestavasset(forvideo:options:resulthandler:)) 取得 AVAsset。

## 回归命令

```powershell
.\.venv\Scripts\python.exe scripts\validate_resource_query_normalization.py
.\.venv\Scripts\python.exe scripts\eval_non_photo_retrieval.py --details
.\.venv\Scripts\python.exe scripts\eval_video_poster_prompt_proxy.py --output reports\video_poster_prompt_proxy_metrics.json
.\.venv\Scripts\python.exe scripts\validate_ios_project.py
```

当前归一化回归包含 17 条 Python 用例；CI 还会直接编译并运行 Swift `SlotNormalizer` 和生产 `CandidateScorer`。Phase 2 已在重新生成的数据上重跑，`with_unknown_cold_start_test` 意图准确率保持 `0.9825`，模型与 App Bundle 的 SHA-256 均为 `BD8ED8B86A5E5BDE4C1475AD5DB1D9EB485857857045BDD6E9D724BCE5EF34E0`。

最终 Swift/Contacts/PhotoKit 编译和 unsigned IPA 必须由固定的 macOS 15 + Xcode 16.3 CI 完成；Windows 本地结果不能替代这一步。
