# 第二阶段小模型训练报告

## 模型

- intent：字符 1-3 gram 线性分类器。
- slots：两个字符级 BIO span 标注器，分别抽取 `share_content` 和 `share_target`，可处理重叠槽位。
- 训练和评估均使用项目 `.venv`，脚本只依赖 Python 标准库。
- 导出格式为 JSON 权重，后续可以在 Swift 端复现同样特征和贪心解码。

## 训练配置

```json
{
  "train_rows": 8800,
  "intent_epochs": 10,
  "tag_epochs": 12,
  "seed": 20260704,
  "datasets": {
    "positive_train": 4000,
    "positive_valid": 1000,
    "positive_cold_start_test": 1000,
    "with_unknown_train": 4800,
    "with_unknown_valid": 1200,
    "with_unknown_cold_start_test": 1200
  }
}
```

## 总体指标

| dataset                      | rows | intent_acc | content_exact | target_exact | keyword_exact | complete_exact | avg_ms   | p95_ms |
| ---------------------------- | ---- | ---------- | ------------- | ------------ | ------------- | -------------- | -------- | ------ |
| positive_train               | 4000 | 1.0        | 0.994         | 0.99925      | 0.9975        | 0.99325        | 0.964319 | 1.3231 |
| positive_valid               | 1000 | 1.0        | 0.992         | 0.994        | 0.992         | 0.986          | 1.021658 | 1.4308 |
| positive_cold_start_test     | 1000 | 0.998      | 0.993         | 0.971        | 0.975         | 0.965          | 1.020151 | 1.3926 |
| with_unknown_train           | 4800 | 1.0        | 0.9875        | 0.996        | 0.9905        | 0.986458       | 0.779142 | 1.099  |
| with_unknown_valid           | 1200 | 0.984167   | 0.971         | 0.974        | 0.97          | 0.94           | 0.757568 | 1.0386 |
| with_unknown_cold_start_test | 1200 | 0.9825     | 0.928         | 0.974        | 0.94          | 0.915833       | 0.785621 | 1.1023 |

## 模型体积与加载

```json
{
  "model_path": "models\\tiny_intent_slot_model.json",
  "model_size_bytes": 1138480,
  "model_size_mb": 1.085739,
  "train_time_ms": 85974.7865,
  "load_time_ms": 125.4953,
  "peak_load_memory_kb": 14223.65
}
```

## 失败样例

```json
{
  "with_unknown_valid": [
    {
      "input": "给所有人都发一份",
      "true": {
        "intent": "unknown",
        "slots": {}
      },
      "pred": {
        "intent": "file",
        "slots": {
          "share_target": "都"
        }
      }
    },
    {
      "input": "帮我把那个文件移动到哥哥的文件夹里",
      "true": {
        "intent": "unknown",
        "slots": {}
      },
      "pred": {
        "intent": "folder",
        "slots": {
          "share_content": "那个文件",
          "share_target": "哥哥的文件夹里"
        }
      }
    },
    {
      "input": "把下载的文件夹传给卧室平板",
      "true": {
        "intent": "folder",
        "slots": {
          "share_content": "下载的文件夹",
          "share_target": "卧室平板"
        }
      },
      "pred": {
        "intent": "folder",
        "slots": {
          "share_target": "卧室平板"
        }
      }
    },
    {
      "input": "把相册里的最新照片分享给家里电脑",
      "true": {
        "intent": "photo",
        "slots": {
          "share_content": "相册里的最新照片",
          "share_target": "家里电脑"
        }
      },
      "pred": {
        "intent": "photo",
        "slots": {
          "share_target": "家里电脑"
        }
      }
    },
    {
      "input": "把相册里的最新照片分享给班主任",
      "true": {
        "intent": "photo",
        "slots": {
          "share_content": "相册里的最新照片",
          "share_target": "班主任"
        }
      },
      "pred": {
        "intent": "photo",
        "slots": {
          "share_target": "班主任"
        }
      }
    },
    {
      "input": "把下载的文件传给前台小刘",
      "true": {
        "intent": "file",
        "slots": {
          "share_content": "下载的文件",
          "share_target": "前台小刘"
        }
      },
      "pred": {
        "intent": "file",
        "slots": {
          "share_target": "前台小刘"
        }
      }
    },
    {
      "input": "把下载的文件传给黄老师",
      "true": {
        "intent": "file",
        "slots": {
          "share_content": "下载的文件",
          "share_target": "黄老师"
        }
      },
      "pred": {
        "intent": "file",
        "slots": {
          "share_target": "黄老师"
        }
      }
    },
    {
      "input": "明天再传这个文件给王瑞冰的项目群吧",
      "true": {
        "intent": "unknown",
        "slots": {}
      },
      "pred": {
        "intent": "file",
        "slots": {
          "share_content": "明天再传这个文件",
          "share_target": "王瑞冰的项目群"
        }
      }
    },
    {
      "input": "把这个压缩包重命名一下",
      "true": {
        "intent": "unknown",
        "slots": {}
      },
      "pred": {
        "intent": "file",
        "slots": {}
      }
    },
    {
      "input": "把下载的文件传给周律师",
      "true": {
        "intent": "file",
        "slots": {
          "share_content": "下载的文件",
          "share_target": "周律师"
        }
      },
      "pred": {
        "intent": "file",
        "slots": {
          "share_target": "周律师"
        }
      }
    },
    {
      "input": "今天天气不错",
      "true": {
        "intent": "unknown",
        "slots": {}
      },
      "pred": {
        "intent": "photo",
        "slots": {}
      }
    },
    {
      "input": "把这个文件夹发给何医生",
      "true": {
        "intent": "folder",
        "slots": {
          "share_content": "这个文件夹",
          "share_target": "何医生"
        }
      },
      "pred": {
        "intent": "folder",
        "slots": {
          "share_target": "何医生"
        }
      }
    },
    {
      "input": "给离线文件夹发个问候",
      "true": {
        "intent": "unknown",
        "slots": {}
      },
      "pred": {
        "intent": "folder",
        "slots": {}
      }
    },
    {
      "input": "等一会儿再分享这个视频给班主任",
      "true": {
        "intent": "unknown",
        "slots": {}
      },
      "pred": {
        "intent": "video",
        "slots": {
          "share_content": "这个视频",
          "share_target": "班主任"
        }
      }
    },
    {
      "input": "下载好的文件发给小王",
      "true": {
        "intent": "file",
        "slots": {
          "share_content": "下载好的文件",
          "share_target": "小王"
        }
      },
      "pred": {
        "intent": "file",
        "slots": {
          "share_content": "下载好的文件"
        }
      }
    },
    {
      "input": "把相册里的最新照片分享给这台设备",
      "true": {
        "intent": "photo",
        "slots": {
          "share_content": "相册里的最新照片",
          "share_target": "这台设备"
        }
      },
      "pred": {
        "intent": "photo",
        "slots": {
          "share_target": "这台设备"
        }
      }
    },
    {
      "input": "老客户名片直接发给阿明",
      "true": {
        "intent": "contact",
        "slots": {
          "share_content": "老客户名片",
          "share_target": "阿明"
        }
      },
      "pred": {
        "intent": "contact",
        "slots": {
          "share_target": "阿明"
        }
      }
    },
    {
      "input": "爷爷，把学习资料文件夹转给我",
      "true": {
        "intent": "folder",
        "slots": {
          "share_content": "学习资料文件夹",
          "share_target": "我"
        }
      },
      "pred": {
        "intent": "folder",
        "slots": {
          "share_content": "学习资料文件夹",
          "share_target": "爷爷"
        }
      }
    },
    {
      "input": "孙会计的档案柜直接发给小何",
      "true": {
        "intent": "folder",
        "slots": {
          "share_content": "孙会计的档案柜",
          "share_target": "小何"
        }
      },
      "pred": {
        "intent": "file",
        "slots": {
          "share_content": "孙会计的档案柜",
          "share_target": "小何"
        }
      }
    },
    {
      "input": "爷爷，把这张旧照片转给他",
      "true": {
        "intent": "photo",
        "slots": {
          "share_content": "这张旧照片",
          "share_target": "他"
        }
      },
      "pred": {
        "intent": "photo",
        "slots": {
          "share_content": "这张旧照片",
          "share_target": "爷爷"
        }
      }
    }
  ],
  "with_unknown_cold_start_test": [
    {
      "input": "把压缩包发给客户经理",
      "true": {
        "intent": "file",
        "slots": {
          "share_content": "压缩包",
          "share_target": "客户经理"
        }
      },
      "pred": {
        "intent": "file",
        "slots": {
          "share_content": "压缩包"
        }
      }
    },
    {
      "input": "你能把那个文件传到这台设备吗？",
      "true": {
        "intent": "unknown",
        "slots": {}
      },
      "pred": {
        "intent": "file",
        "slots": {
          "share_target": "这台设备吗？"
        }
      }
    },
    {
      "input": "存的那张图给上次投屏的电视传过去",
      "true": {
        "intent": "photo",
        "slots": {
          "share_content": "存的那张图",
          "share_target": "上次投屏的电视"
        }
      },
      "pred": {
        "intent": "unknown",
        "slots": {}
      }
    },
    {
      "input": "把压缩包发给财务同事",
      "true": {
        "intent": "file",
        "slots": {
          "share_content": "压缩包",
          "share_target": "财务同事"
        }
      },
      "pred": {
        "intent": "file",
        "slots": {
          "share_content": "压缩包"
        }
      }
    },
    {
      "input": "发条短信给阿明说我会晚点到",
      "true": {
        "intent": "unknown",
        "slots": {}
      },
      "pred": {
        "intent": "file",
        "slots": {
          "share_target": "阿明说我会晚点到"
        }
      }
    },
    {
      "input": "下载好的文件发给物业管家",
      "true": {
        "intent": "file",
        "slots": {
          "share_content": "下载好的文件",
          "share_target": "物业管家"
        }
      },
      "pred": {
        "intent": "file",
        "slots": {
          "share_content": "下载好的文件"
        }
      }
    },
    {
      "input": "把相册里的最新照片分享给客户王总",
      "true": {
        "intent": "photo",
        "slots": {
          "share_content": "相册里的最新照片",
          "share_target": "客户王总"
        }
      },
      "pred": {
        "intent": "photo",
        "slots": {
          "share_target": "客户王总"
        }
      }
    },
    {
      "input": "下载好的文件发给卧室电脑",
      "true": {
        "intent": "file",
        "slots": {
          "share_content": "下载好的文件",
          "share_target": "卧室电脑"
        }
      },
      "pred": {
        "intent": "file",
        "slots": {
          "share_content": "下载好的文件"
        }
      }
    },
    {
      "input": "帮我把这个文档打印三份",
      "true": {
        "intent": "unknown",
        "slots": {}
      },
      "pred": {
        "intent": "file",
        "slots": {
          "share_content": "这个文档"
        }
      }
    },
    {
      "input": "把视频压缩了再保存到电脑",
      "true": {
        "intent": "unknown",
        "slots": {}
      },
      "pred": {
        "intent": "video",
        "slots": {
          "share_content": "视频压缩了再"
        }
      }
    },
    {
      "input": "我把那张照片删了，不用发了。",
      "true": {
        "intent": "unknown",
        "slots": {}
      },
      "pred": {
        "intent": "photo",
        "slots": {
          "share_content": "那张照片"
        }
      }
    },
    {
      "input": "把下载的文件传给家里那台平板",
      "true": {
        "intent": "file",
        "slots": {
          "share_content": "下载的文件",
          "share_target": "家里那台平板"
        }
      },
      "pred": {
        "intent": "file",
        "slots": {
          "share_target": "家里那台平板"
        }
      }
    },
    {
      "input": "下载好的文件发给那台备用机",
      "true": {
        "intent": "file",
        "slots": {
          "share_content": "下载好的文件",
          "share_target": "那台备用机"
        }
      },
      "pred": {
        "intent": "file",
        "slots": {
          "share_content": "下载好的文件"
        }
      }
    },
    {
      "input": "把相册里的最新照片分享给刘师傅",
      "true": {
        "intent": "photo",
        "slots": {
          "share_content": "相册里的最新照片",
          "share_target": "刘师傅"
        }
      },
      "pred": {
        "intent": "photo",
        "slots": {
          "share_target": "刘师傅"
        }
      }
    },
    {
      "input": "把下载的文件传给技术支持",
      "true": {
        "intent": "file",
        "slots": {
          "share_content": "下载的文件",
          "share_target": "技术支持"
        }
      },
      "pred": {
        "intent": "file",
        "slots": {
          "share_target": "技术支持"
        }
      }
    },
    {
      "input": "把相册里的最新照片分享给小宝",
      "true": {
        "intent": "photo",
        "slots": {
          "share_content": "相册里的最新照片",
          "share_target": "小宝"
        }
      },
      "pred": {
        "intent": "photo",
        "slots": {
          "share_target": "小宝"
        }
      }
    },
    {
      "input": "我想把这张图设为壁纸",
      "true": {
        "intent": "unknown",
        "slots": {}
      },
      "pred": {
        "intent": "photo",
        "slots": {}
      }
    },
    {
      "input": "老公，把这个目录发过去",
      "true": {
        "intent": "folder",
        "slots": {
          "share_content": "这个目录",
          "share_target": "老公"
        }
      },
      "pred": {
        "intent": "folder",
        "slots": {
          "share_content": "这个目录"
        }
      }
    },
    {
      "input": "老公，把这个Excel表格发过去",
      "true": {
        "intent": "file",
        "slots": {
          "share_content": "这个Excel表格",
          "share_target": "老公"
        }
      },
      "pred": {
        "intent": "file",
        "slots": {
          "share_content": "这个Excel表格"
        }
      }
    },
    {
      "input": "妈妈，把这张全家福发给你",
      "true": {
        "intent": "photo",
        "slots": {
          "share_content": "这张全家福",
          "share_target": "妈妈"
        }
      },
      "pred": {
        "intent": "contact",
        "slots": {
          "share_content": "这张全家福",
          "share_target": "妈妈"
        }
      }
    }
  ]
}
```

## 说明

- `complete_output_exact` 要求 intent、`share_content`、`share_target` 全部完全匹配。
- `search_keyword_exact` 先用第一阶段相同标准化逻辑生成检索关键词，再和标注侧对齐。
- Python 延迟仅作为开发机基线；最终指标仍需要在 iPhone 真机侧用 App 统计。
