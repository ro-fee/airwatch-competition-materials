# 空域电波哨兵

这是一个基于 PyQt5、pyqtgraph 和 PyTorch 的无线电/RF 信号处理桌面项目。仓库当前保留“智能信号处理原型验证系统”的信号可视化、五类识别、GAN 信号生成、模型轻量化对比和动态频谱瀑布能力，并正在改造为：

> 面向低空无人机的端侧智能频谱感知与未知信号预警平台

项目目标是参加全球校园人工智能算法精英大赛“算法创新赛道—AI+软件创新”。现有程序可作为原型基线运行；真实无人机数据、统一训练工程、多任务/开放集算法和参赛版四页产品闭环仍需按改造任务书实施。

## 参赛改造方向

- 用 DroneRF 等公开真实数据建立可追溯、无泄漏的数据基线。
- 完成 CNN、ResNet18、TCN 基线和时域/时频双分支多任务模型。
- 增加低 SNR 感知、开放集未知设备告警和知识蒸馏轻量化。
- 将现有四个页签改造为频谱态势、无人机识别、鲁棒性实验和端侧评测闭环。
- 所有指标由可复现实验生成，禁止使用合成演示数据、随机数或硬编码结果声明性能。

完整项目口径和实施材料见 [docs/README.md](docs/README.md)。

## 一键运行

Windows 下直接双击项目根目录中的：

```text
一键运行.bat
```

脚本会自动：

1. 切换到项目根目录，避免相对路径失效；
2. 优先查找当前 Conda、`~/anaconda3`、`~/miniconda3` 或 PATH 中的 Python；
3. 检查 PyQt5、PyTorch、NumPy、SciPy、pyqtgraph 等核心运行依赖；
4. 设置 PyQt5/pyqtgraph 后端并启动 `main.py`；
5. 启动失败时保留错误信息和安装提示。

也可以在 PowerShell 中运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_app.ps1
```

只检查环境、不启动界面：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_app.ps1 -CheckOnly
```

## 项目目录

```text
software/
├─ main.py                   # 当前主入口
├─ main2.py                  # 历史兼容入口
├─ newwindow_.py             # 主界面代码（PlotWidget 版本）
├─ MainWindowui.ui           # Qt Designer 源文件
├─ MainWindowui.py           # Qt UI 生成文件
├─ ui_theme.py               # 全局深色主题
├─ waterfall_plotter.py      # 动态频谱瀑布控制器
├─ co.py                     # 运行时颜色表
├─ models/                   # 网络结构、正式权重和历史权重
│  └─ legacy_weights/        # 未被当前主程序引用的旧权重
├─ data/                     # 演示与流程测试数据
├─ result/                   # 特征图、生成信号等运行输出
├─ features/                 # 已有特征图资源
├─ tests/                    # 自动化测试与冒烟测试
├─ tools/                    # 测试数据生成等维护工具
│  └─ legacy/                # 历史实验/数据整理脚本
├─ scripts/                  # 启动和回归脚本
├─ docs/                     # 项目说明、参赛方案、验收规范、任务书与历史材料
├─ launcher/                 # 桌面会话启动器
├─ logs/                     # 历史运行日志
├─ requirements-runtime.txt  # 推荐的精简运行依赖
├─ requirements.txt          # 原始完整环境依赖
└─ 一键运行.bat              # Windows 一键入口
```

## 常用命令

安装推荐运行依赖：

```powershell
python -m pip install -r requirements-runtime.txt
```

直接启动：

```powershell
python main.py
```

完整回归：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_tests.ps1
```

单独运行：

```powershell
python -m unittest tests/test_waterfall.py -q
python tests/smoke_test.py
python tests/smoke_test.py --models
python -m compileall -q .
```

重新生成测试数据：

```powershell
python tools/generate_test_data.py
```

重新生成 Qt UI：

```powershell
pyuic5 MainWindowui.ui -o MainWindowui.py
```

> `newwindow_.py` 是当前 `main.py` 使用的手工调整版界面。重新生成 `MainWindowui.py` 不会覆盖它。

## 文档

- `docs/README.md`：全部文字材料的统一索引与维护状态。
- `docs/项目说明.md`：项目领域、现有功能、软件结构与可信边界。
- `docs/参赛改造方案.md`：参赛主题、数据、算法、产品闭环和指标体系。
- `docs/开发与验收规范.md`：数据、实验、软件、测试和材料规范。
- `docs/空域电波哨兵_项目改造任务书.docx`：8 阶段、64 项任务的执行计划。
- `docs/优化报告.md`：参赛改造前已完成优化的历史基线。
- `docs/软件使用步骤.docx`：原型版软件操作说明，后续随参赛版更新。
- `data/README_测试数据.md`：测试数据结构和适用边界。

## 注意事项

- 请始终从项目根目录启动，或使用一键脚本；模型和数据路径仍有部分采用相对路径。
- `data/` 中的合成数据用于流程验证，不代表真实业务分布，不能用于评价模型业务准确率。
- 当前仓库尚未形成完整的参赛训练与评估工程；不要把计划中的能力描述为已经完成。
- 当前界面中的“信号统计内部一致性”不是真实 FID，轻量化真实对比目前仅支持信号个体识别。
- `EMD-signal`（导入名 `PyEMD`）是 HHT 功能的可选依赖，缺少时其他功能仍可正常使用。
- `tools/legacy/` 中脚本保留了原开发机绝对路径，仅供追溯，不属于当前运行链路。


## Windows 打包

项目已加入 `assets/airwatch.ico` 图标和 `packaging/airwatch_onedir.spec`。首次打包采用 `onedir`，避免把所有依赖压成难排查的单文件。正式发布构建默认执行清洁分析、逐文件校验和包内模型 Workflow 自检：

```powershell
powershell -ExecutionPolicy Bypass -File packaging/build_onedir.ps1
```

日常前端/代码联调可使用增量开发构建，复用 PyInstaller 分析缓存并跳过耗时的包内模型 Workflow 自检，但仍执行资源、哈希和禁止依赖校验：

```powershell
powershell -ExecutionPolicy Bypass -File packaging/build_onedir.ps1 -Mode Dev
```

开发构建不能作为交付证据；准备发布时必须重新运行默认 `Release` 模式。

构建脚本会自动生成便携包说明、`release.json`、SHA-256 和 `runtime-resources.json`，执行逐文件验证，并从刚生成的可执行文件在 CPU 上完成所有已发布模型角色的加载/有限值前向自检，报告写入 `build/package-runtime-smoke.json`。该自检不代表准确率评估。运行时资源由活动模型槽位与冻结契约派生，不整棵复制训练产物。可单独重复验证已构建的包：

```powershell
powershell -ExecutionPolicy Bypass -File packaging/validate_portable_release.ps1
```

目标电脑仍需进行完整人工验收。
