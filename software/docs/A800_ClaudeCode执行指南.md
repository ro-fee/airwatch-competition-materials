# AirWatch A800 V2 上传与执行指南

这份交付用于在共享 Ubuntu 22.04 服务器上训练和评估无人机射频控制源模型。以包内 `bundle-status.json` 的 `status=ready_for_upload` 和完整性校验为本地准备就绪依据。服务器上的 Linux 安装、CUDA 运行和正式训练仍需实测；当前没有新模型性能结论。

## 1 上传什么

上传 `airwatch-a800-20261002.tar` 和同名 `.tar.sha256` 校验文件，放到 `/root/lyh_workspace/`。原始 ZIP/7z 本身已经压缩，外层使用不再压缩的 TAR，降低打包和解包时间。最终精确字节数、SHA-256 见交付旁的校验文件；包内 `bundle-files.json` 列出每个只读文件。

服务器执行：

```bash
cd /root/lyh_workspace
sha256sum -c airwatch-a800-20261002.tar.sha256
tar -xf airwatch-a800-20261002.tar
cd airwatch-a800
cat bundle-status.json
```

如果已经存在同名 `airwatch-a800` 目录，先确认它是否含上一次训练结果；不要覆盖已有运行目录。包中不带本机 Conda 环境，不带桌面安装包。

## 2 给 Claude Code 的完整启动提示词

将下面文字交给服务器上的 Claude Code：

```text
你在 /root/lyh_workspace/airwatch-a800 工作。请先阅读 START_HERE.md、AGENTS.md 和 MODEL_TRAINING_PLAN.md，然后按顺序执行已经准备好的训练包。

先检查 bundle-status.json、SHA256SUMS 和 bundle-files.json。执行 bash scripts/bootstrap_server.sh 完成独立 Linux 环境安装和 CPU 验收。不要在 base 或其他人的环境里安装依赖，不要修改显卡驱动。所有必需测试必须实际执行，缺文件和跳过测试不能算通过。

这张 A800 80GB 是共享资源，需要排队。没有用户或管理员明确给出的 GPU 分配以及本次剩余可用分钟数时，只完成 CPU 阶段，报告等待分配。不要根据 GPU 空闲自行开训，也不要终止其他进程。取得分配后，按本指南的 train 入口启动或恢复。训练器会按固定的 12 次实验执行，不自行更改数据、模型、步数或 batch。

正式训练全部完成且状态有效后，使用同样获配门的 finalize 入口完成验证选型、阈值校准、冻结、保留测试、GPU 基准和模型导出。测试集和未知源只在冻结后评估。VTI 已完成解密与元数据审计，但与冻结输入协议不兼容，本轮仅保留原始归档与审计，不适配、不训练、不宣称外测通过。

每步检查退出码和 outputs/ 下的机器状态。退出码 75 表示等待分配或时段临近结束，下一次明确获配后使用相同入口恢复；不能擅自延长时段。失败不能通过删状态、重算哈希、缩短预算、改指标或跳过样本来掩盖。只有协议已定义的恢复路径可自动使用。需要改变科学协议时先报告，不自行用测试结果调参。

完成后运行 export_results.sh，交回完整模型契约、权重、环境、证据和实测报告。区分工程 smoke、复用的开发验证结果和独立保留测试结果。不能保证新结构一定超过 TCN，负结果也必须保留。
```

## 3 排队期间先做 CPU 准备

```bash
cd /root/lyh_workspace/airwatch-a800
bash scripts/bootstrap_server.sh
```

脚本只创建专用环境 `/root/lyh_workspace/envs/airwatch-a800`，使用 Python 3.10.21 和哈希锁定的 Linux wheel。若该路径存在但没有本包的归属记录，或规格不符，会停止并保留它。安装中断后可重跑同一脚本；不要删除别人的环境。

锁文件为 46 个包的候选 Linux 依赖闭包，Linux 条件依赖已经显式解析。实际安装后还要核对每个版本、运行 `pip check` 和完整 CPU 检查。网络连通不等于 wheel 安装成功。关键报告：

- `outputs/evidence/environment/installed-environment.json`
- `outputs/evidence/environment/pip-freeze.txt`
- `outputs/evidence/environment/cpu-checks.json`
- 失败时的 `outputs/evidence/environment/cpu-checks-failed.json`

只想再次执行 CPU 验收时：

```bash
bash scripts/run_cpu_checks.sh
```

CPU 检查核对 115 个训练录制、24 个验证录制、全部物化文件哈希和保留数据清单。它不对保留信号运行模型。CPU 与数据加载默认只使用少量线程/worker，遵守服务器资源管理要求。

## 4 取得 GPU 后启动或恢复

先记录管理员/调度器给出的分配来源。下面的 `ACTUAL_REMAINING_MINUTES` 必须替换为启动此命令时的真实剩余分钟数，且大于 5；不得自行填入预计训练时间。

```bash
export AIRWATCH_ALLOCATION_SOURCE='填写真实分配来源或作业号'
bash scripts/run_allocated_gpu.sh --phase train --gpu 0 --session-minutes ACTUAL_REMAINING_MINUTES --allocation-confirmed
```

入口会核验 A800 80GB、GPU UUID 和可见计算进程，执行训练数据短跑、显存/吞吐预检和两种结构的 CUDA 断点恢复对照，再依次运行 A/B/C/D 的三个种子。若仍有他人的计算进程则拒绝。主机或容器可见性有限时，显卡进程检查不能替代管理员的分配。

每次获配后都用这个入口。它读取已有 `outputs/runs/<run_id>/` 并核验身份后恢复。保留最后 5 分钟用于保存和清理；INT/TERM 在完成当前优化步后保存。不要使用 `kill -9`，除非服务器管理者要求紧急终止；强杀只能回退到最近有效保存点。

查看进度：

```bash
find outputs/runs -name run-status.json -print -exec cat {} \;
cat outputs/progress/allocation-status.json
```

第一次 GPU 预检会记录实际步耗时与峰值显存，据此估算本机剩余预算。当前没有可信的总训练小时数，不预先承诺几小时完成。

## 5 本轮固定实验

| 组 | 模型 | 每个训练录制的窗口数 | 双视图 JS |
|---|---|---:|---:|
| A | TCN | 32 | 0 |
| B | TCN | 256 | 0 |
| C | 多尺度 TCN | 256 | 0 |
| D | 多尺度 TCN | 256 | 逐步升至 0.1 |

每组种子为 20260909、20260910、20260911；每次固定 9,200 个优化步，40 个逻辑 epoch，每个逻辑 epoch 230 步。每步 128 个锚点，两个独立增强视图合并为一次前向，共 256 个输入。A 每个逻辑 epoch 使用 8 次独立排列，使四组曝光与优化预算一致。BF16 训练，FP32 验证和部署。

训练扩展只覆盖原来的 115 个训练录制：新集合 29,440 窗口；原验证集合 768 窗口保持不变。测试为 24 个保留录制，未知源为四个来源共 204 个录制。MAT 文件独立不代表物理采集会话独立，这一限制必须保留。

检查点选择、三种开放集分数、验收条件、失败判定和统计方法见 `MODEL_TRAINING_PLAN.md`；机器参数以 `training/configs/a800_v2/matrix.json` 为准。旧 V4 只作为单列参考，不参与 V2 组别选型。

## 6 完成训练后的冻结评估

必须在新的或仍然有效的明确分配时段内运行，分钟数仍为启动时的真实剩余值：

```bash
bash scripts/run_allocated_gpu.sh --phase finalize --gpu 0 --session-minutes ACTUAL_REMAINING_MINUTES --allocation-confirmed
```

顺序是完整验证、选型、训练原型/验证阈值校准、冻结、保留已知/未知评估、GPU 基准、模型导出。运行未完成或失败、checkpoint 损坏、数据/代码变化、冻结校验失败时，不会放行保留测试。结束前可保留完成的评估分片，在下一次获配后恢复。

阈值只使用已知验证录制确定；主要分数为余弦原型相似度，MSP/Energy 单列对照。验证集仅 24 个录制，目标 95% 接受率对应至少 23/24，报告实际达到的比例。冻结测试不能反过来决定模型或阈值。

VTI 审计已经核对密码、6 个内层包、全部预处理 MAT 头与 CSV 行列数，并抽样读取两频段原始元数据。实测采样间隔对应 150 MS/s，当前模型输入为 100 MS/s，且预处理矩阵和标签语义不同。本轮交付 `data/raw/VTI_DroneSET.7z` 与 `manifests/vti-audit.json`；没有在包内保存密码，不需要服务器为主训练再次解密 VTI。

## 7 交回结果

全部计算完成后，下面命令只校验并打包已有结果：

```bash
bash scripts/export_results.sh /root/lyh_workspace/airwatch-a800-results.tar.gz
```

需要带回完整 `outputs/release/` 与 `outputs/evidence/`，包括权重、标签、预处理参数、原型/阈值、环境、源码哈希、JSON/CSV 指标、原始计时和失败记录。只拿一个 `.pt` 文件不能接入桌面程序。

A800 上的时延只代表这台 A800；目标电脑的 NVIDIA 型号尚未确定。后续在目标电脑复测，并单独接入桌面推理。本轮不修改现有 V4 桌面运行契约。

## 8 故障与边界

| 现象 | 处理 |
|---|---|
| 退出 75 | 保存状态，等待下一次明确分配；不循环抢卡 |
| 未知旧环境/环境版本漂移 | 停止安装，保留环境，确认正确专用路径 |
| 完整性或恢复身份不符 | 保留原文件，检查传输/版本；不能重写哈希绕过 |
| GPU OOM/NaN/算子不确定性 | 保留诊断，停止正式队列；统一修订协议后才能重启新实验 |
| 缺数据/测试跳过 | 视为准备失败，补齐后重新检查 |
| 新模型未胜出 | 保留预注册对照并报告负结果 |

本包用于授权的被动接收和离线分析。三类标签是射频控制源，不是通用无人机存在检测或飞行状态识别。所有性能必须由评估代码生成，不能把 demo 数据或工程短跑当成业务成绩。
