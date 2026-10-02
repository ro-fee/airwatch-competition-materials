# KU Leuven Drone RF Dataset 数据登记

本目录保存 KU Leuven Research Data Repository 的 `Drone RF Dataset` V1.0 来源审计。固定 DOI 为 `10.48804/HZRVNZ`，官方许可证为 CC BY-NC 4.0。

- `official-api-response.json`：2026-09-08 从 KU Leuven Dataverse 官方 API 保存的完整 V1.0 元数据响应。
- `file-inventory.json`：由审计代码生成的 19 个公开文件清单、官方 MD5、字节数和兼容性门禁。
- `source-documentation/`：从官方文件接口下载并按官方 MD5 验证的 README 与两个 MATLAB 脚本。
- `download-log.json`：元数据、说明文件及首个大型数据包的下载与校验记录。
- `sjrc-pro-archive-audit.json`：`SJRC_pro.zip` 的官方 MD5、本地 SHA-256、51 个 MAT 成员和未解压目录清单。
- `sjrc-pro-sample-content-audit.json`：一个 MAT 成员的 HDF5 变量、形状、复数存储和有限值抽样审计。
- `sjrc-pro-window-reader-acceptance.json`：由受测试读取器生成的真实样本窗口验收，绑定源文件 SHA-256、读取区间、输出形状和运行时版本。
- `sjrc-pro-manifest-v1/`：从 ZIP 直接生成的 51 条成员级录制清单、1,632 行固定窗口计划、验证报告和哈希索引。
- `frysky-download-verification.json`：首个论文已知源包 `Frysky.zip` 的官方大小、MD5 和本地 SHA-256 下载证据。
- `frysky-manifest-v1/`：51 个 Frysky MAT 成员的结构、成员 SHA-256、1,632 行固定窗口计划和哈希索引；当前仍未分配 split。
- `spektrum-dx4e-download-verification.json`：论文已知源包 `Spektrum_DX4e.zip` 的官方大小、MD5 和本地 SHA-256 下载证据。
- `spektrum-dx4e-manifest-v1/`：71 个 Spektrum DX4e MAT 成员的结构、成员 SHA-256、2,272 行固定窗口计划和哈希索引；当前仍未分配 split。
- `mini2-rc-download-verification.json`：论文已知源包 `mini2RC.zip` 的官方大小、MD5 和本地 SHA-256 下载证据。
- `mini2-rc-manifest-v1/`：71 个 DJI Mini 2 RC MAT 成员的结构、成员 SHA-256、2,272 行固定窗口计划和哈希索引；当前仍未分配 split。
- `known-split-v1/`：三个已知源的冻结划分证据；115/24/24 个成员分配到训练/验证/测试，两个边界各丢弃 5 个完整 MAT，共 30 个 guard。
- `known-preprocessing-acceptance-v1.json`：固定 IQ 预处理 `ku-leuven-iq-dc-rms-v1` 的真实数据验收；三个已知源各读取一个已注册训练窗口，验证 `[2,4096] float32`、去直流和复数 RMS 归一化。该文件不是性能证据。
- `known-materialization-verification-v1.json`：D 盘物化训练/验证数组的注册和哈希。共 3,680/768 个窗口，只含 train/validation；test 和 unknown 均未物化。
- `acquisition-semantics-review.json`：对官方说明、DE-FEND 论文和相关原始论文的语义核查，确认 SJRC Pro 为 RC、在论文协议中属于未知源，同时记录会话映射仍未公开。
- `compatibility-semantics-correction.json`：纠正早期把 DroneRF L/H 写成 2.4/5.8 GHz 的错误；L/H 实为 2.4 GHz 下/上半带。
- `external-unknown-protocol-validation-v2.json`：自动校验整个 SJRC 包被锁为外部未知类测试，且未准许训练、验证、阈值选择或模型选择；V1 仅保留为被纠正的历史证据。

官方 V1.0 共 19 个文件、46,696,930,089 字节，其中 16 个 ZIP 数据包。数据为 100 MSamples/s、中心频率 2.44 GHz 的复数 IQ，使用 USRP X310 在半电波暗室采集，MATLAB v7.3 格式保存。官方 README 列出 9 类无人机或遥控器信号。

已知源 `Frysky.zip`、`Spektrum_DX4e.zip` 和 `mini2RC.zip` 均已通过官方字节数、MD5 与本地 SHA-256 校验。三包分别含 51、71 和 71 个 `1×10,000,000` 复数 IQ MAT 成员，全部完成 HDF5 结构与成员哈希审计，并分别生成 1,632、2,272 和 2,272 个不重叠窗口计划。原始窗口计划保持 `split=unassigned`，实际划分由独立哈希证据绑定。连续训练、验证、测试块之间各留 5 个完整 MAT 隔离带，但官方没有证明 MAT 之间是独立物理会话，不得宣称跨会话泛化。固定预处理只做逐通道去直流和逐窗口复数 RMS 归一化，已知源的训练/验证开发链已按冻结配置执行；known test 与 NineEagles、WlToys、Q205、SJRC Pro 四个论文未知源仍全部隔离。

## 当前准入结论

本数据集当前只允许已知源训练/验证开发，**不具备独立性能结论或外部未知评估资格**。首个 `SJRC_pro.zip` 已下载到 D 盘独立数据目录，1,444,331,879 字节与官方 MD5 一致；ZIP 包含 51 个连续编号 MAT 文件，总解压大小 1,502,241,518 字节。51 个成员现已逐一在 ZIP 内完成 HDF5 元数据检查和 SHA-256 计算，全部为 `1×10,000,000` 的 float64 `real/imag`，每条对应 0.1 秒。分块读取器已在真实样本中段读取 4,096 点，输出只读 `[2,4096] float32` I/Q（32,768 字节），未一次载入整段录制。剩余门禁为：

1. 51 条 MAT 文件边界和固定窗口计划已建立，但官方资料没有证明每个 MAT 都是独立物理采集会话；源清单的 `split` 保持 `unassigned`，禁止据文件边界切分训练、验证和测试集；整个归档已在独立协议中锁为 `external_unknown_test`；
2. DroneRF 训练链已能显式只取 2.4 GHz 下半带，但该通道是 40 MS/s 时域实幅值；本数据的两个数值分量是 100 MS/s、单一 2.44 GHz 复数信号的 I/Q。频段接近或形状可调整都不代表语义兼容，禁止直接送入现有模型生成“跨数据集”指标。

CC BY-NC 4.0 允许符合非商业条件的使用和改编，但正式参赛前仍需核对比赛作品授权条款；原始数据默认不随代码、安装包、答辩附件或公开作品再分发。

## 下一步固定顺序

1. 已完成：下载并验证 `SJRC_pro.zip`、只读列出51个成员、抽取一个MAT核验变量和复数表示；
2. 已完成：为 MATLAB v7.3 增加受测试的分块读取器，单次默认最多读取 1,048,576 点，并输出只读 `[2,N] float32` I/Q；
3. 已完成：从 ZIP 只读建立 51 条成员级录制清单和 1,632 行固定、不重叠窗口计划，所有产物由哈希索引绑定；
4. 已完成：论文证据确认 `SJRC pro RC` 是 DE-FEND 的 unknown source；整个归档锁为外部未知类测试，不允许参与训练、验证、阈值或模型选择；
5. 已完成：依据 DroneRF 原始论文纠正 L/H 为 2.4 GHz 下/上半带；数据接口、CNN、ResNet18 与 TCN 支持带身份记录的下半带单通道选择，本机一轮训练冒烟通过且未读取测试集；
6. 已完成：下载并全成员审计已知源 Frysky 和 Spektrum DX4e，两者均保持未划分、禁止训练状态；
7. 已完成：下载并审计第三个已知源 `mini2RC.zip`；三个已知源共 193 个 MAT 成员和 6,176 个窗口计划均已由哈希证据绑定；
8. 已完成：冻结粗粒度连续成员块划分，边界之间保留 5 个 MAT 隔离带；
9. 已完成：冻结单窗口 IQ 去直流+复数 RMS 归一化协议，并用三个真实训练窗口验收，全程未读取测试或未知类数值；
10. 已完成：物化并哈希校验 3,680 个训练窗口与 768 个验证窗口；专用入口拒绝 test，未读取未知类；
11. 已完成：冻结每成员 32 窗口均值 softmax 聚合和已知验证集 95% 接受率阈值规则；RTX 4060 一轮 TCN 工程冒烟已成功，未读取 test/unknown，不作为性能结论；
12. 已完成：固定 A800 上 TCN/ResNet18 × 3 种子的 6 次模型选择矩阵，全部 dry-run 通过；相关配置、入口和进度保持不变，当前暂停执行；
13. 已完成：在本机 RTX 4060 按独立冻结协议完成 TCN/ResNet18 × 3 种子的 6 次训练；仅使用 115 个训练成员和 24 个验证成员，test/unknown 均未读取；
14. 已完成：六个最佳检查点的验证成员级 Macro-F1 均为 `1.0`，说明 24 成员指标已经饱和，不能作为参赛测试成绩。事后细粒度诊断中，TCN 最佳检查点窗口级 Macro-F1 三种子均值为 `0.7750±0.0229`，ResNet18 为 `0.6632±0.0651`；该诊断只用于冻结下一轮开发协议；
15. 已完成：按 `training/configs/ku_leuven_tcn_local_validation_robustness_v1.json` 在本机执行 40 个已知验证条件（clean、9 档复数 AWGN×3 重复、6 档 CFO、2 档多径×3 重复），逐条件保存三个 TCN 和概率集成的指标、逐窗/成员概率以及配对 bootstrap 描述性区间。证据位于 `artifacts/evidence/uav/ku_leuven/local_development/ku_leuven_tcn_validation_robustness_v1.json`，同名前缀包含 CSV、PNG、SVG。集成在 0/5 dB 的成员 Macro-F1 均为 0.5556，在 mild/severe 多径的重复均值为 0.6535/0.7028；-20～0 dB 平均为 0.3115，低于单模型种子/重复均值 0.3411，所以不能声称集成普遍提高抗噪性。多径等级的效果不保证单调；结果仅描述固定相位、窗口零填充和归一化下的数字压力。
16. 已完成：按 `training/configs/ku_leuven_augmented_local_protocol_v1.json` 完成三种子噪声/多径增强训练，并复用全部 40 个验证压力条件。机器生成对比 `artifacts/evidence/uav/ku_leuven/local_development/ku_leuven_augmented_local_comparison_v1.json` 显示预注册开发验收通过：单模型种子/重复平均 clean 成员 Macro-F1 保持 1.0，0/5/10 dB 平均从 0.6677 提高至 0.7895，多径平均从 0.6433 提高至 0.8006。集成在 -5 dB 出现退化，不能据此声明所有噪声条件均改善。验证集已被重复用于开发，结果不是独立测试证据。
17. 已完成：从哈希绑定的预测 CSV 生成 `ku_leuven_noise_error_diagnosis_v1.json`。增强集成在 -5 dB 的 72 次成员/重复预测全部落到 DJI Mini 2，45 次错误中有 18 次至少一个单模型正确，27 次三个单模型均错误；错误平均置信度为 0.5340，没有置信度≥0.9的错误。72 次来自同一24个验证成员的3次扰动重复，不能当作72个独立录制。该诊断定位了输出塌缩与集成丢失正确成员判断的现象，尚未证明具体训练机制的因果关系。
18. 已完成：执行仅把训练 AWGN 档位从 `0/5/10/15/20 dB` 改为 `-5/0/5/10/15/20 dB` 的三种子受控实验，其余训练分支比例、数据、模型、优化器、检查点选择与 40 条件验证矩阵不变。`ku_leuven_strong_noise_local_comparison_v2.json` 的预注册综合验收通过：单模型 `-5/0/5 dB` 平均从 0.5861 提高到 0.7013，多径从 0.8006 提高到 0.8579，clean 保持 1.0。但分档结果否定了“加入 -5 dB 即修复 -5 dB 塌缩”的假设：-5 dB 单模型均值从 0.3738 降至 0.3253，概率集成仍为 0.1818；综合收益主要来自 0 dB（0.4789→0.7787）和 5 dB（0.9056→1.0000）。因此 V2 只能作为中等噪声/多径开发候选，不能宣称解决极低信噪比问题或作为独立泛化证据。
19. 已完成：把 V2 `seed=20260910` TCN 作为三已知源开发演示候选接入软件。运行契约绑定权重、配置、模型定义、预处理和推理代码哈希；Qt 页面支持 CPU 后台识别、进度、取消和失败回传。本机演示样例从 validation 导出到仓库外，元数据明确禁止把它作为性能证据或随作品分发。运行时阈值为空、开放集状态为 `not_evaluated`。
20. 已完成：V3 只改变检查点选择，要求 clean 成员 Macro-F1 至少 `0.97`，再优先最大化三次 `-5 dB` 重复的最差值。`-5 dB` 三种子/三重复均值从 V2 的 `0.3253` 提高到 `0.5854`，但最差值仍为 `0.1818`，因此预注册验收失败。首次运行暴露的守门前早停错误已修复，未完成文件保留在 `artifacts/failed_attempts/ku_leuven_v3_pre_guard_fix_20260909/`，不用于任何结论。
21. 已完成：V4 保留 V3 选择规则，只将 AWGN 分支内 `-5 dB` 权重从 `1/6` 提高到 `1/2`，使其占全部训练抽样 `12.5%`。相对 V3，`-5 dB` 均值为 `0.8569`（`+0.2716`），最差种子/重复为 `0.6308`（`+0.4490`）；clean 保持 `1.0`，`0/5 dB` 均值和多径均上升，五项预注册验收全部通过。该结果只来自重复使用的 24 成员已知验证集和合成扰动，不是独立、跨来源或物理 SNR 证据。
22. 已完成：生成 V4 `seed=20260910` 开发运行契约并替换软件默认无人机模型。该种子是 40 个已重复使用的开发条件上事后综合均值领先者，只作已知源演示候选。契约开放集状态仍为 `not_evaluated`，不发布已知/未知判断。
23. 下一步：固定 V4 的开放集分数和已知验证接受率校准，在模型、阈值和未知协议全部冻结后，才能一次性打开 known test/unknown 生成未知拒识率、AUROC 与 FPR95；不得据该次结果回调方法或阈值。

审计入口：

```powershell
E:\CondaEnvs\rofee-bearing\python.exe -m airwatch.data.ku_leuven_audit --output-root datasets\uav\ku_leuven_drone_rf
```

该命令拒绝覆盖已有审计包；复核新版本时必须使用新的输出目录，不能改写 V1.0 证据。

窗口读取验收入口：

```powershell
E:\CondaEnvs\rofee-bearing\python.exe -m airwatch.data.matlab_v73_iq_audit --path D:\AirWatch_Datasets\ku_leuven_drone_rf\audit_sample\SJRC_pro\sjrc__0.mat --start-sample 5000000 --window-size 4096 --sample-rate-hz 100000000 --output datasets\uav\ku_leuven_drone_rf\sjrc-pro-window-reader-acceptance.json
```

该入口同样拒绝覆盖已有证据。MATLAB v7.3 数据准备依赖单独记录在 `requirements-uav-data.txt`，不强加给桌面运行时。

录制清单与窗口计划入口：

```powershell
E:\CondaEnvs\rofee-bearing\python.exe -m airwatch.data.ku_leuven_manifest --archive D:\AirWatch_Datasets\ku_leuven_drone_rf\source\SJRC_pro.zip --output-dir datasets\uav\ku_leuven_drone_rf\sjrc-pro-manifest-v1 --window-length 4096 --windows-per-recording 32 --seed 20260908
```

该命令直接检查 ZIP 成员，不批量解压，并拒绝覆盖同名证据包。

当前外部未知类协议位于 `training/configs/ku_leuven_sjrc_external_unknown_protocol_v2.json`。窗口级结果只允许作为中间值，正式结果必须先聚合到 MAT 成员，再汇总整个归档；物理会话独立性未知必须在报告中明示。
