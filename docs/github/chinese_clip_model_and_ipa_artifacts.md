# Chinese-CLIP 模型与 IPA 制品链路

## 约束

Chinese-CLIP RN50 的官方 PyTorch checkpoint 超过普通 Git 单文件限制，转换后的 Core ML 包也可能包含超过 100 MiB 的权重文件。因此仓库只保存：

- 固定模型 revision、下载地址、字节数和 SHA-256 的 manifest；
- 下载、转换和校验脚本；
- Xcode 工程中的资源引用。

以下文件必须保持为生成物，不能提交到普通 Git：

- `external_models/pretrained/chinese_clip_rn50/clip_cn_rn50.pt`
- `external_models/converted_ios/chinese_clip_rn50_fp16/*.mlpackage`
- `ios_app/IntentResourceDemo/Resources/ChineseCLIP/*.mlpackage`

CI 首先运行 `scripts/ci/check_no_large_git_files.sh`。任何达到 100 MiB 的普通 Git 跟踪文件都会使构建立即失败。

## GitHub Actions 构建

`.github/workflows/build-ios-unsigned-ipa.yml` 按以下顺序执行：

1. 创建项目根目录下的 `.venv`。
2. 运行 `scripts/download_chinese_clip_rn50.py`，从 manifest 指定的固定官方 revision 下载 checkpoint 并验证 SHA-256。
3. 使用 `scripts/requirements/chinese_clip_coreml.txt` 中固定的依赖运行 `scripts/convert_chinese_clip_rn50_coreml.py`。
4. 缓存已校验的 checkpoint 和转换后的 Core ML 包。
5. 将两个 Core ML 包复制到 Xcode 资源目录，并用 `--require-generated-models` 验证工程和实际模型包。
6. 构建未签名 IPA，并对 IPA 运行 ZIP 完整性检查后生成：
   - `IntentResourceDemo-unsigned.ipa`
   - `IntentResourceDemo-unsigned.ipa.sha256`
   - `ipa-build.json`

`ipa-build.json` 包含 App 版本、构建号、模型标识、模型 checkpoint/source revision、checkpoint SHA-256、Git commit、文件字节数和 IPA SHA-256。Actions artifact 使用零压缩，因为 IPA 本身已经是 ZIP。

## Rolling Release

RN50 实验分支和 `main` 的成功构建会更新专用 prerelease：

- [IPA](https://github.com/ztybuaa/ios_app/releases/download/chinese-clip-rn50-fp16-latest/IntentResourceDemo-unsigned.ipa)
- [SHA-256](https://github.com/ztybuaa/ios_app/releases/download/chinese-clip-rn50-fp16-latest/IntentResourceDemo-unsigned.ipa.sha256)
- [构建信息](https://github.com/ztybuaa/ios_app/releases/download/chinese-clip-rn50-fp16-latest/ipa-build.json)

该 Release 仅用于最新测试版，不覆盖现有稳定 IPA。每次使用都必须以 `ipa-build.json` 和 `.sha256` 为准，不能仅凭固定 URL 判断版本。

## 电脑下载并通过 Wi-Fi 提供给手机

电脑保持 VPN 连接以访问 GitHub，然后在项目根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fetch_and_serve_latest_ipa.ps1
```

脚本会：

1. 下载 Release 的构建信息和 checksum，并要求两者一致。
2. 使用 `curl --continue-at -` 断点续传 IPA。
3. 验证文件大小和 SHA-256；失败时清除本次损坏的 partial 文件并完整重试一次。
4. 选择当前活动的 WLAN/LAN IPv4 地址，避开常见 VPN 虚拟网卡。
5. 在 `8000` 到 `8010` 之间选择空闲端口，启动 `scripts/serve_ipa.ps1`。
6. 自动验证 `HEAD 200` 和 `Range GET 206`，最后输出手机可访问的 HTTP URL。

手机和电脑必须连接同一个 Wi-Fi。手机自身断网后重连同一 Wi-Fi 通常不影响；电脑断网重连后地址可能变化，需要重新运行上述命令生成新链接。下载过程中任一设备断网时，签名软件可以通过 Range 请求续传。
