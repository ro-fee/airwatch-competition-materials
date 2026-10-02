# 测试目录

- `test_waterfall.py`：动态瀑布组件和主窗口集成测试，共 21 项。
- `test_recognition_workbench.py`：三业务互斥与无人机禁用、大图区与底部操作条、真实频谱/特征切换、轴承拒绝与日志隔离、切页不取消轴承/t-SNE 失效、异常与模型提示清理（7 项）。
- `test_bearing_preview.py`：轴承 MAT 通道/清单采样率复用、只读频谱数值、未知/冲突采样率、歧义/NaN/预览上限、真实 Normal_2 通道、后台线程、最新请求优先、取消/关闭清理（8 项）。
- `test_uav_intake_workflow.py`、`test_uav_main_integration.py`：无人机只读输入快照、完整记录质量门、显式采样率、预览/推理同源、按钮就绪门和结构化窗口进度。
  轴承图首轮实现对应 `airwatch/ui/bearing_plot_panel.py` 与 `airwatch/workflows/bearing_preview.py`；不修改模型契约或诊断归一化。
- 真实闭环样本结果保留在 `logs/real_bearing_closed_loop_summary.json`；它是四个样本的功能验收摘要，不是准确率报告。
  轴承图首轮实现对应 `airwatch/ui/bearing_plot_panel.py` 与 `airwatch/workflows/bearing_preview.py`；不修改模型契约或诊断归一化。
- `test_signal_workbench.py`：主页大画布比例、紧凑切换、真实频谱/小波、异步结果双槽与图片比例/滚动、换文件清理、异常提示和专家分析/瀑布切换保护（7 项）。不代表三业务融合界面全部接入。
- `test_waterfall_migration.py`：瀑布新旧计算对照、统一 IQ 输入、图像坐标、时钟/缓冲顺序、兼容入口及错误清理，9 项。
- `smoke_test.py`：无界面启动、核心绘图和可选模型权重冒烟测试。
- `test_home_features.py`：主页专家分析的选区快照不可漂移、非法 HHT 请求复核、工作线程计算与 UI 线程渲染边界（3 项）。
- `test_hht_views.py`：HHT 可选依赖失败、输入/单位、统一主页后台生命周期、测试替身渲染和 Qt 入口（9 项）；不代表真实 EMD 后端验收通过。
- `test_hht_real_backend.py`：真实 EMD-signal 双频/扫频/趋势分解、重构、重复性、8192 点及 Qt 后台出图（4 项）。未装可选依赖时跳过，跳过不能算真实后端验收。
- `../scripts/verify_hht_replacement.py`：7 组合成信号的可复现 HHT 数值检查，使用 `--output <JSON路径>` 保存证据；不是识别准确率或性能基准。可选后端版本见 `requirements-hht.txt`。
- `test_jr_views.py`：J/R 旧公式对照、坐标顺序、极端幅值/零窗、窗口/尾部、无磁盘绘图和真实 Qt 入口（7 项）。
- `test_bispectrum_views.py`：双谱矩公式、单位/尾部处理、输入校验、内存绘图、主页选区和错误清理（6 项）。
- `test_wavelet_views.py`：小波重构/时间轴、输入校验、内存绘图、主页选区、交替显示与错误清理（5 项）。
- `test_tsne_background.py`：真实 sklearn 小样本、快照/文件映射、Qt 后台执行、取消/异常/旧结果拒绝/关闭收尾（8 项）。
- `test_feature_views.py`：模型特征曲线/索引域谱、文件窗口映射、输入校验、坐标语义、缓存清理，以及“中间层特征”选择器到内存 renderer 的真实入口（8 项）；不包括 t-SNE。
- `test_constellation_integration.py`：星座图输入来源、几何坐标、解析兼容、生成类别/样本联动、切图和错误清理（10 项）。
- `test_historical_generation.py`、`test_generation_background.py`：历史生成请求预算、分批推理、按类偏离分数、唯一会话原子提交、取消/失败暂存清理、Qt 后台线程、主窗口发布和用户可写默认目录（9 项）。
- `test_historical_comparison_input.py`、`test_historical_model_comparison.py`、`test_historical_comparison_background.py`：历史双模型比较清单的 SHA-256/输入契约校验、禁止把演示标签提升为真值、双模型后台推理、描述性一致率、准确率 Evidence 门、原子发布、取消/失败清理及页面状态隔离（12 项）。
- `test_runtime_resources.py`：运行时最小资源闭包、模型槽位预留、源码/打包路径复载、发布清单与禁止依赖、PyInstaller 排除项缓存稳定性，以及未变化资源清单不刷新时间戳（14 项）。
- `test_spectrogram_integration.py`：静态时频数值契约、IQ/实信号、页面参数隔离、像素中心、颜色范围、错误清图及切图回归。smoke 逐项检查波形、频谱和时频图的实际数组。

新增图形专项：`test_signal_transforms.py`（数据变换）、`test_plot_registry.py`（图形注册）和 `test_spectrum_integration.py`（真实 Qt 频谱集成及历史切片回归）。

从项目根目录使用专用 Conda 环境执行：

```powershell
& 'E:\CondaEnvs\rofee-bearing\python.exe' -m unittest discover -s tests -p 'test_*.py' -q
& 'E:\CondaEnvs\rofee-bearing\python.exe' tests/smoke_test.py --models
```

或运行 `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_tests.ps1` 完成图形专项、瀑布测试、模型冒烟和全项目编译检查（不等同于完整 unittest discover）。

脚本固定使用上述专用环境；环境不存在就报错，不回退到其他 Python。此路径仅用于当前开发机的测试，不是打包后软件的运行依赖。

通用识别维护链审计（只读、直接穿过 Data / Inference / Workflow Interface；合成输入，不是准确率评估）：`E:\CondaEnvs\rofee-bearing\python.exe tests/audit_general_recognition.py --output logs/general_recognition_audit.json`
