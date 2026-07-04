# 第一阶段数据整理报告

## 结论

- 六份原始 JSON 均已成功解析，`output` 字段均为合法 JSON 字符串。
- 原始监督信号包含 `intent`、`share_content`、`share_target`；`unknown` 样本的 `slots` 为空。
- 标准化数据额外生成 `normalized_slots`，保留 `resource_phrase`，并派生 `search_keyword`、`target`、`qualifiers`。
- 对 `这张照片`、`这个文件` 这类泛指资源，`search_keyword` 显式置为 `null`，依赖 `selection_hint` 或上下文资源选择，不虚构检索关键词。

## 数据集统计

| file                                           | suite         | split           | rows | intent_counts                                                        | keyword_rows                                            | generic_rows                                            |
| ---------------------------------------------- | ------------- | --------------- | ---- | -------------------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------- |
| train_intent_slots(1).json                     | positive_only | train           | 4000 | contact:800, file:800, folder:800, photo:800, video:800              | contact:664, file:752, folder:792, photo:776, video:768 | contact:136, file:48, folder:8, photo:24, video:32      |
| valid_intent_slots(1).json                     | positive_only | valid           | 1000 | contact:200, file:200, folder:200, photo:200, video:200              | contact:166, file:188, folder:198, photo:194, video:192 | contact:34, file:12, folder:2, photo:6, video:8         |
| cold_start_test_intent_slots(1).json           | positive_only | cold_start_test | 1000 | contact:200, file:200, folder:200, photo:200, video:200              | contact:166, file:188, folder:198, photo:194, video:192 | contact:34, file:12, folder:2, photo:6, video:8         |
| train_intent_slots_with_unknown.json           | with_unknown  | train           | 4800 | contact:800, file:800, folder:800, photo:800, unknown:800, video:800 | contact:628, file:265, folder:667, photo:394, video:354 | contact:172, file:535, folder:133, photo:406, video:446 |
| valid_intent_slots_with_unknown(1).json        | with_unknown  | valid           | 1200 | contact:200, file:200, folder:200, photo:200, unknown:200, video:200 | contact:147, file:67, folder:170, photo:106, video:94   | contact:53, file:133, folder:30, photo:94, video:106    |
| cold_start_test_intent_slots_with_unknown.json | with_unknown  | cold_start_test | 1200 | contact:200, file:200, folder:200, photo:200, unknown:200, video:200 | contact:159, file:75, folder:176, photo:104, video:96   | contact:41, file:125, folder:24, photo:96, video:104    |

## 总体统计

```json
{
  "rows": 13200,
  "intent_counts": {
    "photo": 2400,
    "video": 2400,
    "file": 2400,
    "folder": 2400,
    "contact": 2400,
    "unknown": 1200
  },
  "slot_key_counts": {
    "share_content": 12000,
    "share_target": 12000
  },
  "search_keyword_counts": {
    "photo": 1768,
    "video": 1696,
    "file": 1535,
    "folder": 2201,
    "contact": 1930
  },
  "generic_resource_phrase_counts": {
    "photo": 632,
    "video": 704,
    "file": 865,
    "folder": 199,
    "contact": 470
  }
}
```

## 标准化槽位示例

| intent  | input               | normalized_slots                                                                                                                                                                                                             |
| ------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| photo   | 把这张照片发给班主任          | {"resource_type": "图片/照片", "resource_phrase": "这张照片", "search_keyword": null, "target": "班主任", "target_keyword": "班主任", "qualifiers": {"time": [], "format": [], "selection_hint": ["current_selection"]}}                   |
| video   | 把这个视频发给研发群          | {"resource_type": "视频", "resource_phrase": "这个视频", "search_keyword": null, "target": "研发群", "target_keyword": "研发群", "qualifiers": {"time": [], "format": [], "selection_hint": ["current_selection"]}}                      |
| file    | 把这个文件发给家里那台电脑       | {"resource_type": "文件", "resource_phrase": "这个文件", "search_keyword": null, "target": "家里那台电脑", "target_keyword": "家里那台电脑", "qualifiers": {"time": [], "format": [], "selection_hint": ["current_selection"]}}                |
| folder  | 把这个文件夹发给技术支持        | {"resource_type": "文件夹/目录", "resource_phrase": "这个文件夹", "search_keyword": null, "target": "技术支持", "target_keyword": "技术支持", "qualifiers": {"time": [], "format": [], "selection_hint": ["current_selection"]}}               |
| contact | 把这个联系人发给phone_6973  | {"resource_type": "联系人/联系方式", "resource_phrase": "这个联系人", "search_keyword": null, "target": "phone_6973", "target_keyword": "phone_6973", "qualifiers": {"time": [], "format": [], "selection_hint": ["current_selection"]}} |
| unknown | 用QQ把旅行视频发给北区销售群怎么操作 | {}                                                                                                                                                                                                                           |

## 高频资源短语

### contact
| share_content | count |
| ------------- | ----- |
| 这个联系人         | 58    |
| 王医生的联系方式      | 29    |
| 我的电子名片        | 25    |
| 今天新增的联系人      | 24    |
| 我的联系方式        | 24    |
| 这张名片          | 23    |
| 张经理的联系方式      | 22    |
| 李老师的联系方式      | 22    |
| 电子名片          | 22    |
| VCF名片         | 22    |
| 合作方名片         | 22    |
| 我的手机号         | 21    |

### file
| share_content | count |
| ------------- | ----- |
| 文件            | 135   |
| 压缩包           | 81    |
| 这个文件          | 75    |
| Excel表格       | 54    |
| 文档            | 43    |
| PDF文档         | 30    |
| 上个月的文件        | 29    |
| 下载好的文件        | 26    |
| Word文档        | 25    |
| 十个文件          | 25    |
| 项目计划书         | 23    |
| PPT文件         | 23    |

### folder
| share_content | count |
| ------------- | ----- |
| 这个文件夹         | 64    |
| 项目文件夹         | 50    |
| 备份文件夹         | 45    |
| 照片文件夹         | 40    |
| 工作文件夹         | 34    |
| 素材归档目录        | 33    |
| 那个文件夹         | 33    |
| 设计文件夹         | 31    |
| 共享文件夹         | 26    |
| 资料夹           | 25    |
| 素材文件夹         | 23    |
| 音频素材文件夹       | 22    |

### photo
| share_content | count |
| ------------- | ----- |
| 这张照片          | 124   |
| 图片            | 91    |
| 照片            | 90    |
| 刚才拍的照片        | 68    |
| 截图            | 61    |
| 今天拍的照片        | 58    |
| 相册里的最新照片      | 49    |
| 相册里的第一张照片     | 39    |
| 昨天拍的照片        | 37    |
| 相册            | 37    |
| 上周拍的照片        | 30    |
| 这张截图          | 28    |

### video
| share_content | count |
| ------------- | ----- |
| 视频            | 254   |
| 这个视频          | 131   |
| MP4视频         | 48    |
| MOV视频         | 36    |
| 录制的视频         | 34    |
| 这段视频          | 33    |
| 活动视频          | 32    |
| 今天拍的视频        | 30    |
| 五个视频          | 29    |
| 旅行时拍的视频       | 29    |
| 昨天的视频         | 28    |
| 所有视频          | 28    |

## 转换逻辑

- `resource_phrase` 直接来自原始 `share_content`，用于训练和可解释展示。
- `search_keyword` 从 `resource_phrase` 中去除 intent 对应资源类型词，例如 `图片/照片/文件夹/联系方式` 等；如果去除后没有有效语义词，则置为 `null`。
- `target` 和 `target_keyword` 直接来自原始 `share_target`，供联系人或设备候选检索使用。
- `qualifiers.time` 抽取 `今天/昨天/最近/刚才/上周/上个月` 等时间限定词。
- `qualifiers.format` 抽取 `PDF/Word/Excel/PPT/PNG/JPG/GIF/MP4/MOV/VCF/vCard` 等格式限定词。
- `qualifiers.selection_hint` 标记 `current_selection`、`recent`、`all` 等资源选择语义。

## 输出文件

- `processed/dataset_manifest.json`：六份数据的 manifest 和统计。
- `processed/datasets/*.jsonl`：每份原始数据对应一份标准化 jsonl。
- `processed/datasets/all.jsonl`：六份数据合并后的标准化 jsonl。

## 下一阶段建议

1. 用 `with_unknown` 套件训练完整的六类 intent + 槽位抽取模型。
2. 用 `positive_only` 套件单独报告五类正向能力，避免 unknown 指标掩盖资源类表现。
3. 将 `share_content/share_target` 转成字符级 BIO 标签，训练端侧可部署的小型槽位抽取模型。
4. 先实现 `unknown` 拦截和 `contact` 检索模块，再接 photo/file/folder/video 模块。
