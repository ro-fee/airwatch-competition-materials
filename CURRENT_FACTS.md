# 当前事实与可引用结论（2026-10-03）

这是材料写作的最新事实入口；早期 Word/Markdown 草稿中“无人机尚未接入”“A800 尚未训练”“测试尚未读取”等语句需要按时间区分，不能直接照抄。数值由回传证据生成，完整映射见 FACTS.json。

| 项目 | 当前事实 | 证据 |
|---|---|---|
| 正式实验 | A/B/C/D × 3 种子，共 12 次完整训练，每次 9,200 步 | evidence/a800-v2/local-integrity-audit.json |
| 已选部署模型 | A-seed20260909，TCN，856,067 参数 | outputs/evidence/selection.json、outputs/release/runtime-contract.json（均在 evidence/a800-v2/ 下） |
| 已知测试 clean | 24 条 MAT 成员，Accuracy=1.0000，Macro-F1=1.0000 | evidence/a800-v2/local-runtime-verification.json |
| 主开放集分数 | cosine；阈值只用已知验证集校准 | evidence/a800-v2/outputs/release/open-set.json |
| 未知测试 | 204 条 MAT 成员、4 个未知来源；AUROC=0.8781 | evidence/a800-v2/outputs/evidence/test/metrics.json |
| 固定阈值表现 | 已知接受率 91.67%；未知拒识率 43.63%；未知误接受率 56.37% | 同上、local-runtime-verification.json |
| FPR95 | 0.5833，为测试 ROC 上的描述性统计，不是重新选择部署阈值 | 同上 |
| 新结构是否胜出 | 否；B/C/D 未通过预注册改进门槛，保留 A | evidence/a800-v2/outputs/evidence/selection.json |
| 桌面状态 | 现有无人机页仍使用 V4；A800 已在本机完成 CPU 验证推理，尚未接入 UI | software/docs/项目说明.md、local-runtime-verification.json |

注意：上表仅为选定部署种子的 clean 主结果，不代表所有扰动条件或三种子均值。三种子/40 条件数据请读取 FACTS.json 的 seed_summary 及原始 metrics.json。绘图应由脚本读取这些文件，禁止手工构造或美化指标。

## 必须同步说明的边界

- 这是三种已知射频控制源分类和已知模型拒识，不能写成通用无人机存在检测、飞行状态识别或可靠低空安防系统。
- 小规模同源 MAT 成员测试不证明物理采集会话独立，不证明跨地域、跨接收机或真实连续监测泛化。
- 合成 AWGN 的 dB 不是物理现场 SNR；未知拒识仍有明显误接受，必须展示失败案例。
- VTI 与当前输入协议不兼容，未形成可用外测结果。
- A800 基准与目标 NVIDIA 电脑不同；服务器记录了共享 GPU 执行变更，不能描述成目标端独占性能。
- D-seed20260910 首次于 1840 步失败后完整重跑，失败记录保留；不能抹去。
- 历史五类模型、GAN、轴承验证与本届 UAV 新增代码必须区分；不得把历史业务能力或论文方法整体声称为本届原创。
- GAN 指标不是 FID；模型一致率不是准确率；多任务、学生模型、最终四页完整闭环仍有计划内容。

## 资料优先级

1. 本页与 FACTS.json 的最新汇总，逐项回查链接中的机器证据。
2. software/docs/项目说明.md、参赛改造方案.md、开发与验收规范.md。
3. software/ 下实际代码及测试。
4. reference/ 中 Word 文本摘录、早期技术报告雏形与历史功能说明仅作结构参考。

当前快照用于私有写作协作，不含全部实验数据和运行权重，不能据此宣称完成了重新训练、完整测试或安装包验收。
