# DroneRF 数据登记

本目录保存 DroneRF V1 的本地数据治理记录。数据集来自 Mendeley Data，DOI 为 `10.17632/f4c2b4n755.1`，许可证为 CC BY 4.0。

- `source/DroneRF-v1.zip`：官方原始归档，只读保存；官方与本地 SHA-256 均为 `f3cd8a1dfe14f51edc40c8012f12cdf34c3d7e1c51b62ef4539505d9d3f32c0d`。
- `packages/DroneRF/`：从官方 ZIP 提取的 23 个嵌套 RAR，只读源包区；CSV 尚未展开。
- `recording-manifest.csv`：由 RAR 成员表自动生成的 227 条 L/H 双子带录制对；L/H 分别是 2.4 GHz 采集带宽的下半段与上半段。
- `split-manifest.csv`：固定种子 `20260908`、按标签分层的录制级训练/验证/测试归属。
- `split-audit.json`：覆盖、重复、缺失、每类分布、源归档与清单校验值的机器生成审计证据。
- `download-log.json`：下载时间、来源、版本、大小和校验记录。
- `source-semantics-review.json`：依据原始数据论文纠正 L/H 物理语义，记录 40 MS/s、时域实幅值以及 2.4 GHz 上下半带结论。
- `processed/dronerf-two-band-raw-windows-v1-w4096-n32-seed20260908/`：经全长计数、选窗和独立落盘复核的 227 个双频段窗口产物，共 7,264 个窗口。

当前状态为“同源录制级基线训练已放行”。每个原始频段成员均完成 10,000,000 点全长计数，选中窗口通过数值和有限值检查；同一 `recording_id` 的窗口不会跨集合。DroneRF V1 未提供独立采集会话、日期或物理设备实例 ID，因此该划分只能支持同源录制级评估，不能被描述为跨会话或跨设备泛化。完整原始数据默认不进入安装包；任何演示样本都必须保留数据集署名、许可链接和修改说明。

重复生成清单与审计：

```powershell
E:\CondaEnvs\rofee-bearing\python.exe -m airwatch.data.prepare_dronerf --package-root datasets\uav\dronerf\packages --output-dir datasets\uav\dronerf --seed 20260908 --source-archive datasets\uav\dronerf\source\DroneRF-v1.zip
```
