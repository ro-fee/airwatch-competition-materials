# 给 Your dot 的启动说明

你负责“空域电波哨兵”的竞赛文字与演示材料。项目所有者已提供这份私有源码与实验证据快照，请基于实际产物完成材料写作，保持事实、计划和结论的区别。

## 阅读顺序

1. `CURRENT_FACTS.md`、`FACTS.json`：最新状态、已选模型、能引用的结果与限制。
2. `WRITING_BRIEF.md`：交付项与验收要求。
3. `software/docs/项目说明.md`、`software/docs/参赛改造方案.md`、`software/docs/开发与验收规范.md`。
4. `software/main.py`、`software/airwatch/`、`software/training/`：核对架构、真实功能和算法实现。
5. `evidence/a800-v2/outputs/evidence/selection.json`、`test/metrics.json`、`test/member-predictions.csv` 与 `outputs/release/runtime-contract.json`。这些路径均以 `evidence/a800-v2/` 为根；不要寻找服务器绝对路径。
6. `reference/` 的任务书和旧报告，只用于补充结构；必须修正过时内容。

## 执行方式

先输出资料理解和证据矩阵，再开始技术报告与 PPT 故事线。缺少赛事模板、时长、页数等信息时集中提出问题，同时先完成不依赖答案的内容。建议的篇幅和页数是工作草案，不是已核实的赛事要求。

所有成品保存到 `materials/`，附可编辑源文件。若当前运行环境能制作 DOCX/PPTX，则同时提交成品与可编辑底稿；如果不能，先交 Markdown 报告、逐页 PPT 文案与图表数据/生成脚本，明确未生成的格式，不伪装成已完成文件。

不要启动训练、修改模型阈值或替换桌面默认模型。不要提交到赛事平台，不要公开仓库。需要访问补充来源时优先官方赛事通知、数据集官网、论文原文和库的官方文档，逐条核对引文，记录访问日期。

## 可以直接发送给 dot 的提示词

> 请基于这个私有仓库完成“空域电波哨兵”的竞赛技术报告、答辩 PPT、作品简介、演示讲稿及答辩问答。先完整阅读 DOT_START_HERE.md、WRITING_BRIEF.md、CURRENT_FACTS.md 和 FACTS.json，再结合 software/ 的实际代码与 evidence/ 的机器证据撰写。按照任务书分阶段交付到 materials/，每个关键论断都要有证据路径；保留负结果、失败记录和未知拒识限制。A800 结果已训练回传，但尚未接入桌面 UI，不得写成已完成部署闭环。若缺赛事模板等必要信息，集中提问并继续独立部分。不要替我投稿、公开仓库、发送外部消息或修改实验数据。
