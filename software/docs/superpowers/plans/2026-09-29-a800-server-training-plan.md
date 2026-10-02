# A800 服务器训练包与无人机模型训练计划书

> 执行器：服务器上的 Claude Code。按任务顺序执行，每项任务用复选框记录结果；所有必要命令、边界和验收规则均应包含在交付包中，不依赖 Codex 插件或本机技能。
>
> 文档状态：实现与交付验收版，2026-10-02。用户已批准准备，并指定服务器执行器为 Claude Code。代码、数据审计和训练协议已落实；本地包是否就绪以 bundle-status.json 与完整性校验为准。尚未在服务器安装、训练或生成正式指标。

**Goal:** 准备一个可整体上传至 Ubuntu 服务器的训练包，利用排队获得的单张 A800 80GB 完成可恢复的模型训练、消融、开放集评估及 GPU 推理模型交付。

**Architecture:** 保留当前桌面软件、TCN V4 权重和旧实验协议。新实验使用独立的 A800 V2 配置、训练器和输出目录，复用已验证的信号预处理、扰动、模型与录制聚合实现。服务器只运行训练与离线评估，最终输出可由桌面软件接入的冻结模型契约。

**Tech Stack:** Ubuntu 22.04 x86_64、独立 Conda Python 3.10.21 环境、PyTorch 2.14.0+cu126、NumPy、SciPy、h5py、scikit-learn、Matplotlib；数据解密工具独立登记。

**Spec:** 本项目的《参赛改造方案》《项目说明》《开发与验收规范》和《空域电波哨兵_项目改造任务书》；本次用户要求是本地准备文件、用户上传、Claude Code 在服务器执行，最终面向 NVIDIA 显卡电脑运行。GPU 使用权须排队取得。

## 1 全局约束

- 本地准备已获用户授权。仅准备上传文件和本地工程验收；服务器 CUDA 工作须另有明确 GPU 分配及真实剩余时段。
- 若任何步骤需要 gcc/g++ 编译，必须在同一 shell 进程中先执行 conda deactivate，再编译；单独另起 shell 的 deactivate 不算满足要求。
- 本机应用和 Python 测试使用 E:\CondaEnvs\rofee-bearing\python.exe；服务器使用本计划规定的独立环境。两者不要混为一谈。
- 服务器主目录拟定为 /root/lyh_workspace/airwatch-a800；环境放在 /root/lyh_workspace/envs/airwatch-a800。运行入口也必须支持整体目录迁移。
- 服务器现有 base 为 Python 3.14.7；本轮实验另建 Python 3.10.21 环境，不改 base 或其他现有环境。
- 用户确认当前 GPU 任务属于别人，需要排队。GPU 暂时空闲不等于获得使用权。Claude Code 不自动终止其他进程、不重置 GPU、不更改 MIG、功率上限或驱动。
- 历史 A800 V1 配置、旧划分、V4 运行契约及其哈希绑定代码保持原样。新模型定义放新文件，训练输出使用新目录。
- 禁止将测试、未知源、guard 成员或 VTI 外部验证数据用于训练、自监督预训练、特征选择、增强选择、检查点选择或阈值选择。
- 所有结果由代码生成 JSON/CSV；配置中的步数、阈值校准目标及验收线是实验设计，不是已经取得的性能。
- 已知类别仅指三种已知射频控制源。当前标签不支持通用无人机存在检测、工作状态识别或多无人机分离；本轮不因增加输出头就宣布这些能力。
- 合成 AWGN 的 dB 为原窗口功率相对新增噪声功率的比例，不称为物理采集 SNR。
- MAT 成员边界已知，但物理采集会话独立性未经证实。多种子和更多窗口不能消除该限制。
- GPU 部署目标已明确；最终电脑的显卡型号和性能要求尚未提供。A800 时延只能作为 A800 的结果。
- 密码不写入公开计划书、命令行参数、普通日志、清单或发布模型包。解密过程读取用户提供的受控输入，失败时只报告解密失败。
- 所有新增代码在 airwatch/ 或 training/ 内实现；scripts/ 仅放运维入口。桌面 UI 改造另行执行，本轮交付推理契约和适配所需代码。
- 仓库当前不是可用 Git checkout。构建时以代码文件 SHA-256 和源快照记录版本；只有确认存在 Git 仓库时才附加 commit，不编造 commit。

## 2 审查重点

1. **共享 GPU 与使用时段**：未取得分配、卡仍有外部进程或时段到期时，应拒绝开训或可恢复退出。对应 T4 的资源门测试。
2. **Linux 迁移与隐藏依赖**：绝对盘符、大小写、动态导入和跳过测试可能造成假通过。对应 T3/T8 的独立目录验收及必须运行的真实数据测试。
3. **训练恢复正确性**：恢复后样本顺序、增强、优化器、学习率、随机数和最佳检查点应衔接。对应 T4 的中断对照测试。
4. **数据泄漏与语义不兼容**：扩大训练覆盖、VTI 格式适配及未知源读取都必须服从清单与冻结门。对应 T1/T2/T6。
5. **选型与报告失真**：相同训练预算、独立测试门、分数方向、聚合单位和实际部署精度应一致。对应 T5/T6/T7。

## 3 已确认环境与准备状态

以下来自用户在 2026-09-29 提供的服务器输出，是当时快照，开训前重新检查。

| 项目 | 已确认值 | 对本计划的影响 |
|---|---|---|
| 系统 | Ubuntu 22.04.4 LTS，x86_64 | 使用 Linux x86_64 安装包 |
| CPU | nproc 输出 128 | 不视为本任务独享 128 核；初始仅 4 个加载 worker |
| 内存 | 251GiB 总计，226GiB available | 适合物化窗口与内存映射；不需整库常驻 |
| 磁盘 | / 分区可用 309G | 足够规划基础包，但提取前按声明展开大小复核 |
| Conda | 26.7.1 | 创建独立 prefix 环境 |
| base Python | 3.14.7 | 不作为本轮训练 Python |
| GPU | A800 80GB PCIe，MIG 关闭 | 单卡训练；不需要分布式训练 |
| 驱动 | 580.159.03 | 支持本轮 cu126 运行路线 |
| GPU 状态 | 33453MiB 占用，利用率 100% | 属于别人的任务，等待分配 |
| 网络 | PyPI 与 PyTorch cu126 索引均 HTTP 200 | 可在线安装依赖；实际 wheel 下载仍需安装时验证 |
| 可用 GPU 时长 | 未知 | 使用可中断、可恢复的任务队列，不承诺训练总天数 |
| 排队系统 | 未确认 | 以管理员/用户明确分配为门；不因主机名猜测 Kubernetes 调度配置 |

本机已读取确认 Python 3.10.21、torch 2.14.0+cu126、numpy 2.2.6、scipy 1.15.3、h5py 3.16.0、scikit-learn 1.7.2。官方 cu126 索引已核实存在 torch 2.14.0 的 cp310 manylinux_2_28_x86_64 wheel。这只证明候选安装包存在，不能替代服务器安装和 CUDA 前向验收。

nvidia-smi 的 CUDA Version 表示驱动支持版本，不表示本机已安装对应版本的 Toolkit。使用官方 PyTorch wheel 时按其依赖安装，不修改服务器驱动，也不要求管理员额外安装 CUDA 13。

## 4 数据范围和上传清单

### 4.1 主训练数据

主线继续使用 KU Leuven V1.0 三个已知源：

| 归档 | 标签 ID | 本地原始归档字节数 |
|---|---:|---:|
| Frysky.zip | 0，frysky | 1707367398 |
| Spektrum_DX4e.zip | 1，spektrum_dx4e | 2026208466 |
| mini2RC.zip | 2，dji_mini2_rc | 2078132999 |

现有只读物化集位于本机 D:/AirWatch_Datasets/ku_leuven_drone_rf/prepared/known-iq-v1。它在项目目录外，必须纳入上传包。

- 训练：115 个成员，3680 个窗口。
- 验证：24 个成员，768 个窗口。
- 保留已知测试：24 个成员，仍封存。
- guard：30 个成员，不参与任何学习和报告。
- 单窗口 float32 [2,4096]，100 MS/s，逐通道去直流及复数 RMS 归一化。
- 每个验证/测试成员固定 32 个窗口，按均值 softmax 概率聚合。
- 所有既有清单与窗口计划原样保留哈希；迁移路径通过新的位置映射记录，不改写原始证据。

### 4.2 未知源

SJRC_pro.zip 已在本机，大小 1444331879 字节。其余三包拟在本地准备时补齐，使用户上传后不必再寻找数据：

| 归档 | 官方文件 ID | 预期字节数 | 预登记 MD5 |
|---|---:|---:|---|
| NineEagles.zip | 154427 | 1491713902 | d08bc6bf7cfa9d267d6f637f58c384a5 |
| Q205.zip | 154428 | 1531075910 | 3506e1e507b6575f6d741e93effdb3f5 |
| wltoys.zip | 154421 | 1513957160 | fb2847bfe3696b835c900fc253ef3f89 |

下载前复核固定数据版本的官方元数据；下载后检查字节数、官方 MD5，并生成本地 SHA-256。仅做完整性、结构、录制边界与固定窗口登记；模型未冻结前不输出预测、分数或类别表现。

### 4.3 VTI_DroneSET

本机 source/VTI_DroneSET.7z 已存在，6277143852 字节；既有 SHA-256 为：

524e038f6577729778dbd31910076880e296b24e3047e4976d7c74a6b7bc9a3a

用户提供的密码已验证成功；内层结构与实际元数据检查范围见 T1 及 manifests/vti-audit.json。当前结论为与冻结表示不兼容，没有运行模型。

VTI 继续作为独立外部验证候选。本轮必须交付内容审计、原始文件身份及兼容性结论；跨来源性能属于条件性交付：

- 有明确的信号表示、采样率/频段、标签和录制边界，且与冻结模型输入兼容，才放行对应外测。
- 不能仅靠“都是两行数组”判断 IQ 与实幅值兼容。
- 如果需要新的重采样、频段转换或派生表示，必须有独立版本的物理语义说明和数值测试；不能从 VTI 的预测成绩挑选转换方式。
- 没有可靠兼容方案时，输出 blocked_incompatible_representation，并保留数据作为后续实验来源；本轮以 KU Leuven 的独立保留测试完成主验收。
- 若将来要把 VTI 改作训练数据，应建立新的数据协议和外部留出来源，不沿用本轮“独立外测”的结论。

### 4.4 文件体积与存储

已补齐并核验全部上述原始归档，合计 18069931566 字节，约 18.07GB 十进制。这是原始压缩归档体积，不是最终上传包或解压峰值。

拟增加的训练覆盖集含 115×256=29440 个窗口；信号数组主体为 29440×2×4096×4=964689920 字节，实际文件还含头部、标签及清单。这是确定性尺寸计算，不是实测产物。

准备阶段先读取归档声明展开大小，估算原始包、提取目录、缓存和原子提交暂存的同时占用，再实际测量。服务器 CPU 准备初始要求可用空间至少 120GiB；任务执行中可用空间低于 20GiB 时停止生成新数据并安全保存进度。若审计得到的展开峰值超过这一预算，先采用逐成员流式读取方案；不得清理他人文件腾空间。

## 5 模型实验设计

### 5.1 训练覆盖扩展

新增 dense-train-v2，仅改变训练成员内的覆盖：

1. 保留每个训练成员原有的 32 个冻结窗口。
2. 在该成员中枚举以 4096 为步长、完整落在录制内的候选窗口。
3. 删除与原 32 窗口有任何重叠的候选。
4. 将剩余候选按起点排序，等分为 224 个非空连续候选组；每组确定性选一个。
5. 随机生成器种子取 SHA-256("dense-v2|20260929|" + recording_id) 前 8 字节的无符号小端整数，使用 NumPy PCG64。
6. 合并为 256 个窗口并按起点排序。任何成员无法满足数量或不重叠约束，则拒绝构建并报告，不减少窗口数或挪用 guard。
7. 只读取 115 个训练成员。验证、测试和 guard 不改变。
8. 同步生成逐窗起止点、源成员 SHA-256、预处理版本和数组行号。

原 v1 Dataset/训练器仍要求 32 窗口，不直接放宽其限制。新训练入口消费 v2 数据契约，并显式允许 train=256、validation=32。

### 5.2 固定矩阵

| 实验 ID | 模型 | 训练覆盖 | 损失 | 相邻比较 |
|---|---|---|---|---|
| A | 原 DroneRFTCN | 每成员 32 窗口 | 双视图监督损失 | A800 V2 对照 |
| B | 原 DroneRFTCN | 每成员 256 窗口 | 同 A | B−A：覆盖 |
| C | MultiScaleRFTCN | 每成员 256 窗口 | 同 A | C−B：结构 |
| D | MultiScaleRFTCN | 每成员 256 窗口 | 监督损失 + 一致性 | D−C：一致性 |

每组固定种子 20260909、20260910、20260911，共 12 次正式训练。

A 是统一新版训练协议下的对照：复用 V4 增强分布，但双视图、训练预算和 batch 已明确版本化。它不能标注为原 V4 训练的完全复现。原 V4 三种子冻结权重作为历史参考随包携带，注明训练条件不同。

### 5.3 输入、模型与损失

所有模型输入 [batch,2,4096]，输出三个类别的 logits，另提供 forward_features() 返回 256 维表示。保持相同预处理和 32 窗口成员级聚合。

MultiScaleRFTCN 的结构决策：

- stem 复用原 TCN 的 Conv1d(2,32,kernel=15,stride=4,padding=7)、BN、ReLU。
- 四个残差块的输出通道为 64/128/128/256，dilation 为 1/2/4/8。
- 每块第一层改为 kernel=3/7/15 的三个并行因果卷积分支，每分支通道为 Cout/4；拼接后 1×1 卷积投影到 Cout，再 BN、ReLU、Dropout(0.1)。
- 每块第二层保持 kernel=5、相同 dilation 的因果卷积和 BN。
- 残差投影、相加、ReLU，以及前两个块后的 MaxPool1d(2)，遵循原 TCN 结构。
- AdaptiveAvgPool1d(1) 得到 256 维表示，Linear(256,3) 分类。
- 参数量和耗时由实现后程序测量，本计划不预报性能。

每个训练锚窗口产生两个独立的 V4 分布增强视图；所有 A/B/C/D 都计算两个视图，避免 D 因额外输入数量获得隐含优势：

- 每个视图：50% clean、25% AWGN、25% 多径。
- AWGN 档位 −5/0/5/10/15/20 dB，权重 5/1/1/1/1/1。
- mild/severe 多径沿用现有冻结参数。
- 增强只来自训练数据；确定性键包含 run_seed、logical_epoch、draw_position、window_id、view_id，重复抽到同一窗口也能得到不同且可恢复的增强。
- 两个视图拼接成一个 batch 前向，使各实验的 BN 使用方式一致。
- 监督损失为两个视图的加权交叉熵均值。类别权重使用训练数据数量倒数并归一化为均值 1。
- A/B/C 的一致性系数为 0。
- D 加入两个视图概率分布的 Jensen–Shannon 散度，两个分支均反传；softmax 与散度在 FP32 中计算，log 输入下限 1e-7。
- D 的一致性系数从 0 在前 230 步线性增至 0.1，之后保持；不在正式矩阵中搜索系数。

### 5.4 训练预算与精度

| 参数 | V2 决策 |
|---|---|
| 锚窗口 batch | 128；两个视图共 256 个输入 |
| 每个逻辑 epoch | 29440 个锚窗口，230 个优化步 |
| 正式预算 | 40 个逻辑 epoch，9200 个优化步/次 |
| 总正式次数 | 12，最多 110400 个优化步 |
| 优化器 | AdamW，lr=0.001，weight_decay=0.0001，betas=(0.9,0.999)，eps=1e-8 |
| 学习率 | 前 230 步线性 warmup 至 0.001，之后 cosine 至 0.00001 |
| 梯度裁剪 | 全局范数 1.0 |
| 精度 | CUDA autocast BF16；参数/优化器状态保持 FP32，BF16 不使用 GradScaler |
| 验证/校准/主推理 | FP32；关闭 TF32，避免引入另一套部署分数 |
| 数据加载 | 4 workers，pin_memory；worker 数可在预检中统一下调并冻结 |
| CPU 限制 | OMP/MKL 线程初始 4，torch intra-op=4、interop=1 |
| 正式早停 | 不启用；各实验具有相同固定预算，用验证集选最佳检查点 |
| CUDA 编译 | 首轮不启用 torch.compile，减少恢复与移植变量 |
| 确定性 | 启动前设置 CUBLAS_WORKSPACE_CONFIG=:4096:8；deterministic algorithms=True、cudnn.benchmark=False，记录不支持算子的真实错误 |
| 日志 | 每 50 步及每个验证周期输出结构化进度 |
| 保存 | 每个逻辑 epoch；收到可处理的终止信号或达到时段边界时在完成当前优化步后保存 |

A 每个逻辑 epoch 对 3680 个训练窗口执行 8 次独立确定性排列；B/C/D 对 29440 个窗口执行一次排列。全部都有 29440 个锚窗口、58880 个前向输入及 230 次参数更新。实际耗时不保证相同，报告同时给出训练步数、样本呈现数及实测 GPU 时间。

BF16 不可用、固定 batch OOM 或存在不可支持的确定性算子时，本轮预检失败；只能在正式矩阵开始前统一修改配置并生成新的协议版本。不能在个别正式实验中悄悄换精度、batch 或算子。

短训练只使用独立 smoke run ID，最多 200 步，不进入正式指标表。正式时间估算来自获配 GPU 后的实测，计算时包括验证和保存；排队等待单独记录。

### 5.5 检查点与模型选择

每 230 步评估 clean 与三个固定重复种子的 −5 dB 已知验证条件，沿用 V4 的开发选择顺序：

1. clean 成员 Macro-F1 至少 0.97 才有资格；
2. 最大化三个 −5 dB 重复中的最差成员 Macro-F1；
3. 再比较重复均值、clean Macro-F1、负 clean window loss；
4. 完全并列时保留更早的检查点。

未出现合格检查点的 run 标记 failed_no_eligible_checkpoint；不放宽守门线。0.97 是既有选择规则；24 成员验证集粒度很粗，应同时保留逐成员预测和窗口指标。

若该 run 已完成全部固定步数，failed_no_eligible_checkpoint 属于已记录的科学负结果，可在冻结清单中明确排除其不存在的合格权重，其余完整实验仍可评估。NaN、OOM 或预算未完成造成的失败不适用此规则；不得混称为模型负结果。若 A 的三种子没有完整合格对照，则停止本轮正式比较并报告，不能自行换一组种子。

各实验三种子全部完成后，在完整的原 40 条件验证矩阵评估最佳检查点。候选要替代 A，需同时满足：

- clean、0/5 dB 均值和多径均值相对 A 的下降均不超过 0.03；
- −5 dB 三种子三重复均值严格高于 A；
- −5 dB 最差种子/重复不低于 A。

在合格 B/C/D 中按 −5 dB 最差值、−5 dB 均值、clean 均值、较少参数量依次选型；都不合格则保留 A 并记录负结果。部署种子固定为 20260909，不根据测试或事后 40 条件排名挑种子；三种子总体结果均须报告。

所有选型仍是重复使用的开发验证集证据。即使开发门通过，独立测试不支持改善时，也必须如实报告，不能将开发收益作为最终成绩。

## 6 开放集与测试冻结

### 6.1 分数和阈值

预注册主方法为余弦类原型分数，MSP 和 Energy 为对照；不根据未知测试结果更换主方法。

- 成员类别预测：32 个窗口 softmax 向量的算术均值取 argmax。
- MSP：上述成员均值概率的最大值。
- Energy 已知性：每窗 logsumexp(logits)，T=1，再取 32 窗均值；数值越高越像已知。
- 原型拟合：只用所有模型共有的原 v1 clean 训练窗口。先对窗口 feature 做 L2 归一化，再按成员平均并再次归一化；按类平均这些成员表示并归一化为类原型。
- 原型已知性：待测成员表示与三个训练类原型的最大余弦相似度。
- 零范数、缺窗、混合标签或非有限分数为数据/模型错误，不用默认值代替。
- 各模型各分数独立校准阈值。对 24 个已知验证成员，取可接受至少 ceil(0.95×24)=23 个成员的最大阈值，判定为 score>=threshold。处理并列分数并记录实际接受率，不能声称恰好 95%。
- 已知验证集很小且已参与开发，校准不代表真实场景可靠性。

### 6.2 冻结门

在读取保留测试数值进行模型评估前，freeze.json 必须完整绑定：

- 所有正式 run 的最终状态、候选和基线最佳检查点哈希；
- 模型定义、训练配置、数据清单、预处理、聚合和评分代码；
- 原型及校准阈值文件；
- 验证选型结果、预定部署种子；
- 测试清单与窗口计划、40 条件协议、评价脚本；
- 软件版本/源码哈希、主推理精度；
- 无测试/未知访问的训练日志和数据访问审计。

这里只是流程隔离，不能把同一 root 用户下的目录与代码检查描述为不可绕过的安全隔离。训练入口拒绝 sealed 数据；原始已知 ZIP 含测试成员，训练读取器仅放行列入训练清单的成员。

测试门校验通过后，Claude Code 可按已批准的执行计划运行评估，不再重复索要逐项确认；模型、阈值、测试协议的修改不属于该自动执行范围。

### 6.3 测试输出

- 已知保留测试：24 成员，每成员 32 窗口；类别指标和原 40 条件压力测试。
- 未知主测试：四个冻结未知源，仅先做 clean 开放集评估。
- 比较全部已完成且已预注册的正式实验；原 V4 权重单列历史参考，披露训练预算差异。
- 主要统计单位是 MAT 成员，不将窗口、重复噪声或随机种子当作新增独立录制。
- 报告成员 Macro-F1、Accuracy、分组混淆矩阵、三种子均值/样本标准差。
- AUROC 约定已知为正类，分数越大越已知。
- FPR95 为测试 ROC 上 TPR>=0.95 的最小 FPR，使用阶梯曲线、无插值；只作描述，不把测试 ROC 的阈值回写部署文件。
- 部署阈值另报实际已知接受率、未知误接受率、未知拒识率和已知样本被拒后的分类覆盖率。
- 95% 区间按类别/来源分层的成员级配对 bootstrap，2000 次，种子 2026090904；说明物理会话相关性未知，因此区间只描述当前成员抽样协议。
- 成功结果原子提交，记录 evaluation-completed.json。重复调用已完成评估应核验后返回已有结果，不覆盖或重新调参。
- 评估中断可按已冻结配置继续剩余分片；失败记录和已完成分片都保留。

## 7 实际上传包结构

~~~text
airwatch-a800/
  START_HERE.md                   Claude Code 执行指南和启动提示词
  AGENTS.md                       服务器协作规则
  MODEL_TRAINING_PLAN.md           本计划的随包快照
  bundle-status.json              本地准备状态，明确服务器尚未实测
  bundle-files.json               每个不可变文件的字节数与 SHA-256
  SHA256SUMS                     校验 bundle-files.json，避免循环哈希
  airwatch/                      数据、模型、扰动、推理及其源码依赖
  training/                      训练、评估、交付、数据准备和配置
  scripts/                       CPU 安装/检查、获配 GPU、结果打包入口
  tests/                         必需测试；缺失、零执行或跳过均拒绝
  environment/                   直接依赖、Linux 候选闭包、解析元数据
  data/raw/                      7 个 KU Leuven ZIP 和 VTI 原始 7z
  data/prepared/known-iq-v1/      原 32 窗口训练/验证数组
  data/prepared/dense-train-v2/   256 窗口训练及原验证数组
  manifests/                     冻结划分、窗口计划、VTI 审计、封存评估索引
  provenance/                    官方来源及历史登记证据
  artifacts/                     保持原路径的 V4 参考权重与证据
  preparation-evidence/          本地独立目录验收证据
  outputs/                       服务器可写目录，不纳入输入包哈希
~~~

原始数据统一存储，访问隔离由固定配置、划分核验和冻结门执行，不以目录名字替代准入检查。V4 的权重和哈希绑定源码保持原相对路径和原字节，旧证据内的本机绝对路径仅作历史来源记录。主训练使用包内相对路径。

保留 training/common.py 所需的小型 CWRU Dataset/BearingCNN 源码依赖，不携带轴承数据、Qt 页面或桌面安装包。最终完整文件闭包由独立目录导入、测试与真实数据入口核验。

## 8 本地准备实现与验收入口

本节记录实际接口。Claude Code 收到就绪包后直接执行第 9 节，不需要重新开发这些模块。最新本地验收数量、文件身份与状态以机器报告为准；源码工作记录在 training/A800_IMPLEMENTATION_CONTRACT.md。下面列出的 CUDA 检查必须留在获配后的服务器阶段。

### T1 VTI 审计

实现：airwatch/data/vti_audit.py。受控工具为隔离安装的 py7zr 1.1.3；密码通过交互输入，未写入包或普通日志。

真实审计已验证外层解密/完整性和 6 个内层包的 SHA-256；检查全部 50 个预处理 MAT 头、4 个 DNN CSV 行列数，并抽查两频段各一个原始 MAT 元数据。原始 Y 形状为 [67107072,1]，XDelta=6.666666666666667e-9，中心频率分别 2.449/5.792 GHz；预处理 Data 为 [670,2048]；CSV 为 2051 行、10050 或 16750 列。没有运行模型或据预测推断标签。

结果统一保存在 manifests/vti-audit.json，compatibility.status=blocked_incompatible_representation。150 MS/s 与本轮 100 MS/s 协议不同，聚合矩阵和多无人机标签没有经过独立适配。本轮外测指标条件性不支持，不阻塞合法 KU Leuven 主实验。原始未抽样内容不宣称已逐采样点检查。

### T2 KU Leuven 数据

实现：training/a800_data.py 的 dense/unknown/sealed 子命令；airwatch/data/ku_leuven_training_v2.py 的 dense_offsets 与 A800Dataset。

V2 数据读取接口 A800Dataset(root, split='train', verify_hashes=True) 返回 (tensor,label,index)，并提供 samples/sample_metadata/recording_ids/data_identity/close。只允许 train/validation。旧 V1 读取器没有放宽。

数据入口在读取原始成员前校验冻结 split SHA、清单索引、原窗口计划和源 ZIP。实物输出为 115×256=29440 个训练窗口，原 3680 窗口逐个保持一致；验证 768 窗口保持原字节。四个未知源共 204 个录制只有元数据与固定 32 窗计划，未做模型预测。保留读取器校验归档路径、成员大小上限、成员 SHA、HDF5 点数与不重叠窗口。

必需测试：tests.test_a800_data、tests.test_a800_data_integrity；实际数据检查由 training.a800_cpu_checks 执行，不因本机盘符不存在而跳过。

### T3 Linux 环境和路径

实现：training/resolve_a800_environment.py、training/a800_environment.py、scripts/bootstrap_server.sh、scripts/run_cpu_checks.sh。

候选 Linux 闭包为 46 个包；按 Linux x86_64、CPython 3.10.21 显式计算 marker/extras，验证 Requires-Python 与 wheel 平台并锁定 URL/SHA-256。少数索引未提供 SHA 时下载对应 wheel 计算摘要，记录来源。此结果不是 Linux 安装成功证明。

服务器使用唯一专用 prefix /root/lyh_workspace/envs/airwatch-a800。bootstrap 在新建环境时保存归属/锁身份；已有未归属或规格不同的环境拒绝修改。安装命令使用 --require-hashes --no-deps，依赖已由闭包完整锁定；随后 pip check、逐包版本核对和 CPU 检查。中断的本包安装可按同一锁继续，已安装环境发生版本漂移则停止。

### T4 恢复与资源门

实现：training/a800_runner.py、training/a800_common.py、scripts/run_allocated_gpu.sh。公开入口为 python -m training.a800_runner --phase train|finalize --gpu INDEX --session-minutes MINUTES --allocation-confirmed；建议始终使用 shell 包装脚本。

内部 run_training 接受配置、训练/验证数据、设备和完整身份；run_matrix(root,allocation,...) 协调固定队列。CLI 在 CUDA 前核对不可变包和专用环境。GPU 门保存实际来源、UUID、起止时间；不抢占、不轮询等空闲。磁盘剩余低于 20 GiB 时拒绝新任务或在可恢复边界暂停。INT/TERM 与时段截止共享停止状态。

CPU 对照覆盖连续运行与中断恢复的权重、优化器、学习率、增强、样本次序、随机状态和最佳检查点一致；损坏最近文件可回退上一有效状态，身份变化拒绝恢复。CPU 路径不初始化 cuDNN。获配后还必须执行两种结构的 CUDA/BF16 固定 batch 短跑、计时和显存预检，再用均衡的三类真实训练小子集比较连续 4 步与 2 步后序列化恢复到 4 步的全部恢复状态。检查不相等或不支持确定性算子时拒绝正式开训。工程目录独立，不能当作模型性能。CUDA 检查仍须服务器实际执行，本地没有代跑。

### T5 模型与双视图目标

实现：airwatch/models/uav_multiscale_tcn.py、training/a800_objective.py。

MultiScaleRFTCN(num_classes=3,in_channels=2) 的 forward 返回 logits，forward_features 返回特征。build_a800_model 支持 tcn/multiscale_tcn。paired_objective(model,view1,view2,labels,js_weight=...,class_weights=...) 在一次前向中合并两视图，返回 loss 与 CE/JS 诊断。JS 用 FP32、两分支均有梯度；增强键绑定 seed、epoch、draw position、window ID 和 view ID。

测试覆盖形状/反传、因果卷积、零系数、视图交换、极端 logits、增强重现和 BN 单前向。机器矩阵固定在 training/configs/a800_v2/matrix.json，不能为单个正式 run 临时改参数。

### T6 冻结评估

实现：training/a800_evaluation.py。validate_run/calibrate_run 接受 config、checkpoint、root 和有效 allocation；freeze_experiment 绑定已完成恢复证据、模型、全部协议、代码、数据和校准结果；evaluate_frozen 只接受通过哈希核验的冻结文件。run_finalization(root,allocation=...) 协调整体流程。

完成状态不能只相信 run-status.json；必须验证最后恢复文件、最佳文件及其 receipt、完整 9200 步和 40 次验证、身份及选择历史。验证/校准缓存绑定 V1 数据字节与评估代码，最终指标能从分片成员预测重新计算。V4 单列参考不参与 A/B/C/D 选型。

测试覆盖分数方向、23/24 校准、原型来源、已知正类 AUROC/FPR95、冻结篡改、分片幂等和伪完成状态拒绝。真实保留测试在服务器冻结后执行，不在本地准备阶段生成指标。

### T7 导出和测量

实现：training/a800_delivery.py、airwatch/inference/uav_a800_contract.py。

benchmark_frozen/export_release 受有效 allocation 控制；load_contract(path,device='cpu') 校验契约、工件及实际导入源码；predict_member(contract,iq_windows,sample_rate_hz) 复用冻结 FP32 预处理与分数。工程测试使用临时模型，不混入正式 evidence。

正式导出包含 model.pt、label-map.json、preprocessing.json、open-set.json、runtime-contract.json、环境、必需源码与指标引用。测量保留原始计时，单窗/32窗、模型/端到端分别报告。CPU package 子命令仅打包已有完成结果；没有 GPU 副作用。

### T8 上传包与验收

实现：training/build_a800_bundle.py 的 stage/index/verify/finalize/package 子命令，及 training.a800_cpu_checks。

staging 先核对必要文件；默认不覆盖冲突源，仅允许明确的准备阶段刷新，已发布包与训练目录拒绝覆盖。不可变清单排除 outputs 和自引用文件；符号链接、越界路径、缺失文件、未登记文件和摘要变化均拒绝。每个必需测试模块必须实际执行，测试报告绑定当时的 bundle index；旧成功、零测试或更新的失败记录不能发布 ready。

独立目录清除 PYTHONPATH 回指后测试，并核对真实数据。通过后固定 bundle-status.json、文件清单和 SHA256SUMS，构建 TAR 与外部 SHA-256。包内 preparation-evidence 保存本地证据，服务器实际环境与正式实验只写 outputs。

## 9 服务器 Claude Code 执行顺序

以下为已实现入口。必须从完成校验的上传包根目录执行，不能把本机工作区当成服务器环境。

### 9.1 排队期间的 CPU 阶段

~~~bash
cd /root/lyh_workspace/airwatch-a800
bash scripts/bootstrap_server.sh
bash scripts/run_cpu_checks.sh
~~~

CPU 阶段核验文件、环境和数据身份；不启动 CUDA 前向、校准、训练、GPU 基准或测试集模型评估。现有 CPU 资源也由多人共享，遵循管理员限制。

### 9.2 获得 GPU 后

run_allocated_gpu.sh 接受执行阶段、GPU index 和本次获配分钟数。用户确认分配后，由 Claude Code 将实际值传入；不预填未知时长。

~~~bash
bash scripts/run_allocated_gpu.sh --phase train --gpu 0 --session-minutes ACTUAL_ALLOCATED_MINUTES --allocation-confirmed
~~~

ACTUAL_ALLOCATED_MINUTES 在执行时必须替换为本次真实获配的正整数分钟数。这是尚未由管理员提供的外部信息，不是允许 Claude Code自行估算的值。

脚本顺序固定为：GPU 资源门 → CUDA smoke → 吞吐/显存预检 → 恢复已有未完成 run → 按 A/B/C/D 及种子顺序执行正式矩阵。时段内无法完成则保存，退出码 75；下一次获配后同命令恢复。获配标志不是自动排队或抢占实现。

### 9.3 正式训练全部结束后

~~~bash
bash scripts/run_allocated_gpu.sh --phase finalize --gpu 0 --session-minutes ACTUAL_ALLOCATED_MINUTES --allocation-confirmed
~~~

该入口调用 T6 的 run_finalization，依次执行 select-and-calibrate、freeze、test、benchmark 和 export；内部 freeze 路径固定为 outputs/evidence/freeze.json。脚本层和 Python 层都执行资源门，不能仅靠文档约束直接评估命令。分配时段结束则保存阶段/分片状态，下一次获配后用同命令恢复。训练矩阵不完整、存在未处理失败或冻结校验不通过时，不得执行 test。

全部计算完成后，scripts/export_results.sh 仅校验并打包已经完成的交付文件，不额外启动 GPU 工作。

## 10 故障处理与任务状态

| 条件 | 必须行为 |
|---|---|
| GPU 未分配或被他人占用 | 记录 waiting_for_allocation，退出 75；不自行轮询抢卡 |
| 使用时段临近结束 | 完成当前优化步、原子保存、释放 CUDA，退出 75 |
| 网络失败 | 有上限重试并保留下载暂存；不替换来源或版本 |
| 文件校验失败 | 拒绝使用该文件，列出路径与期望/实际哈希；不能关闭校验 |
| VTI 密码错误 | 记录 password_verification_failed，不把数据标为可用 |
| VTI 语义不兼容 | 保留审计结果，跳过对应外部指标；不尝试根据预测表现适配 |
| 磁盘不足 | 停止新物化/新 run，保存当前状态；不删除他人文件 |
| CUDA OOM | 记录配置与峰值，停止；预检阶段可统一修订协议，正式阶段不临时更改 batch |
| NaN/Inf | 保存诊断与失败状态，停止 run；不修改指标或跳过坏 batch 后宣称成功 |
| 恢复身份不一致 | 拒绝恢复，保留旧产物；修复后需新 run/新协议或恢复正确输入 |
| 正式预算未跑完 | 标记 incomplete；不静默缩成“完整 12 次实验” |
| 改进未成立 | 报告负结果，依预注册规则保留对照；不追加测试驱动的调参循环 |

只有事先定义的可恢复故障允许自动恢复。科学协议的更改、新数据用途、其他人进程处置或扩大训练范围均需用户重新给出指示。

每个 run 写 run-status.json，状态限定 planned/running/interrupted/completed/failed。实验矩阵进度由这些文件生成，不依赖聊天上下文；每次 Claude Code 续接先校验状态文件与产物，而不是盲目从头训练。

## 11 交付与验收标准

本地准备阶段完成条件：

- VTI 密码验证和内容审计有真实结果，失败则清楚标为阻断或条件性缺项。
- 三个已知源、四个未知源和现有物化数据的所需文件完整，哈希可复核。
- 新训练覆盖、模型、损失、恢复与评估门通过相关测试。
- 上传包可在独立目录运行，服务器关键数据测试没有因 Windows 路径错误被跳过。
- 有精确上传清单、机器检查结果、主计划书和 Claude Code 启动说明。
- 本轮修改未破坏当前桌面运行契约；相关历史回归仍通过。

服务器执行阶段完成条件：

- 独立环境、CPU 检查和获配后的 CUDA smoke 实测通过。
- 12 次正式训练完成固定预算；短跑与失败产物没有混入结果表。
- 验证选型、三种分数校准、模型/代码/协议冻结可追溯。
- 保留已知与未知评估由冻结入口完成，失败和负结果均保留。
- VTI 外测按兼容性结论执行或明确不支持；没有将缺项伪装成通过。
- 导出模型在包内复载通过，服务器基准有原始计时记录。
- 最终报告明确三类已知源、合成扰动、MAT 成员独立性和目标电脑未实测的边界。
- 返回用户 outputs/release、完整 Evidence、配置、环境锁和关键训练日志；不把只有 .pt 的文件当完整交付。

成功标准是获得可复现的比较和可接入的软件模型；不能保证新结构一定超过旧 TCN，更不能保证未知拒识已达到业务要求。

## 12 给 Claude Code 的启动提示词

~~~text
请执行当前目录中已经验收的 AirWatch A800 V2 训练包。先阅读 START_HERE.md、AGENTS.md 和 MODEL_TRAINING_PLAN.md，然后校验根目录 SHA256SUMS、bundle-files.json 与所有必要文件。

用户已准备 Ubuntu 22.04 x86_64、单张 A800 80GB 的共享服务器。当前需要排队；没有明确获配信息时，只运行 CPU 环境和数据检查并报告就绪状态。不得终止别人的任务或根据 GPU 暂时空闲自行开训。

使用 /root/lyh_workspace/envs/airwatch-a800 独立环境。不要在 base 或其他已有环境中安装训练依赖，不要修改驱动。不要依赖 Codex 插件或任何未包含在包中的技能。

获配 GPU 和使用分钟数明确后，按计划执行资源检查、短跑、恢复和固定的 12 次实验。训练仅允许读取训练和验证清单。保持测试、未知数据及 VTI 外测的用途边界。只在模型、阈值、数据和代码全部冻结后执行保留测试。

每一步保存结构化进度与日志；中断后从已验证 checkpoint 恢复。遇到协议未覆盖的模型/数据改动时停止该变更并报告，不能自行扩展实验或依据测试结果调参。禁止编造或手工修改指标。

最后导出完整模型契约、权重、标签、预处理、拒识阈值、环境、JSON/CSV 证据、基准与失败报告。完成情况以机器产物和验收条件为准。
~~~

此提示词在 bundle-status.json 标记 ready_for_upload 且逐文件校验通过后使用。完整上传和运行步骤见 START_HERE.md；本机源文档为 docs/A800_ClaudeCode执行指南.md。

## 13 依据与后续信息

- 项目源文档：docs/项目说明.md、docs/参赛改造方案.md、docs/开发与验收规范.md、docs/空域电波哨兵_项目改造任务书.docx。
- 既有 A800 配置：training/configs/ku_leuven_a800_model_selection_v1.json；只作历史参考，V2 不覆盖它。
- 现有 V4 协议与证据：training/configs/ku_leuven_weighted_worst_noise_local_protocol_v4.json 及 artifacts/evidence/uav/ku_leuven/local_development/ 下对应机器产物。
- [PyTorch 官方 cu126 wheel 索引](https://download.pytorch.org/whl/cu126/torch/)：已核实指定 cp310 Linux x86_64 wheel 条目。
- [NVIDIA 驱动与 CUDA 版本说明](https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html)。
- [NVIDIA CUDA 兼容性](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。
- [KU Leuven V1.0 数据页](https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/HZRVNZ)。
- [VTI_DroneSET V1 数据页](https://data.mendeley.com/datasets/s6tgnnp5n2/1)。

目前不阻碍本地准备但必须在相应阶段补充的信息：管理员实际分配时段与调度方式、最终 NVIDIA 电脑显卡型号和时延需求。VTI 已完成本轮审计并明确当前协议不兼容，未生成外测指标。

本计划自查覆盖了数据身份、Linux 迁移、资源分配、恢复、实验公平性、测试封存及完整模型交付。实现中若必须改变算法或数据协议，应先更新本计划并新建机器协议版本，再启动受影响的正式实验。

