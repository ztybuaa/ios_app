# 用 GitHub Actions 打包 IPA

当前你没有 Mac，但可以让 GitHub Actions 的 macOS runner 编译 Xcode 工程。

## 推荐流程：先生成未签名 IPA，再用手机签名工具签名

你买的全能签/轻松签/万能签这类产品通常需要一个已经编译好的 IPA。  
本工程提供的 GitHub Actions 会生成：

- `IntentResourceDemo-unsigned.ipa`

这个 IPA 不能直接安装，需要在 iPhone 的签名工具里导入并使用你的：

- `.p12` 证书
- p12 密码
- `.mobileprovision` 描述文件

签名后再安装。

## 操作步骤

1. 把整个项目上传到 GitHub 仓库。
2. 打开仓库的 `Actions`。
3. 选择 `Build unsigned iOS IPA`。
4. 点击 `Run workflow`。
5. 等待完成后，在 workflow run 的 `Artifacts` 下载：
   - `IntentResourceDemo-unsigned-ipa`
6. 解压 artifact，得到：
   - `IntentResourceDemo-unsigned.ipa`
7. 把 IPA 传到 iPhone。
8. 在手机签名工具中导入 IPA、P12、描述文件，签名并安装。
9. 到 iPhone：
   - 设置 -> 通用 -> VPN 与设备管理
   - 信任对应证书
10. 打开 App 测试。

## 如果你想让 GitHub Actions 直接生成已签名 IPA

也可以，但需要把证书和描述文件作为 GitHub Secrets 上传。  
这对共享证书或淘宝购买证书风险较高，不建议你把 P12 密码放进 GitHub。

如果你确认要这么做，需要准备：

- `IOS_P12_BASE64`
- `IOS_P12_PASSWORD`
- `IOS_MOBILEPROVISION_BASE64`
- `IOS_TEAM_ID`
- `IOS_BUNDLE_ID`

然后再扩展 workflow 做签名 archive/export。

## 本地 Windows 不能完成的部分

Windows 可以做：

- 数据处理
- 模型训练
- 工程文件生成
- GitHub workflow 准备

Windows 不能直接做：

- 编译 iOS Swift 工程
- 调用 iPhoneOS SDK 产出 `.app`
- 原生 Xcode archive/export

这些步骤必须在 macOS/Xcode 环境中完成，GitHub Actions 的 macOS runner 就是替代方案。

## 失败排查

- Actions 里 `xcodebuild` 失败：看日志里具体 Swift 编译错误。
- 下载的是 zip：GitHub artifact 会先包成 zip，解压后才是 IPA。
- IPA 导入签名工具失败：先确认 zip 已解压，导入的是 `.ipa` 文件。
- 签名后无法安装：检查描述文件是否支持你的设备、Bundle ID 是否匹配、证书是否被吊销。
- 安装后提示无法验证完整性：信任证书，或证书/描述文件已失效。
