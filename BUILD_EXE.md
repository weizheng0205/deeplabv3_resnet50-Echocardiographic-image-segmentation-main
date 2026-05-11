# EchoSeg UI 打包说明

## 1. 准备环境

建议在项目根目录创建独立环境后安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果需要 CUDA 版 PyTorch，请先按本机 CUDA 版本安装 `torch` 和 `torchvision`，再安装其余依赖。

## 2. 编译 exe

推荐使用目录模式，启动更稳定，也更适合 PyTorch：

```powershell
.\build_exe.ps1
```

输出位置：

```text
dist\EchoSegUI\EchoSegUI.exe
```

如果必须生成单文件：

```powershell
.\build_exe.ps1 -OneFile
```

单文件模式体积会很大，首次启动也会更慢。

## 3. 运行方式

把 `data`、`output` 或模型权重放在 exe 可访问的位置，打开程序后在界面中选择：

- 权重文件：例如 `output\segmentation\deeplabv3_resnet50_random\best.pt`
- 数据集目录：例如 `data\EchoNet-Dynamic`
- 输出目录：例如 `output\segmentation\deeplabv3_resnet50_random`

点击“加载已有结果”可直接查看已经生成的 `test_dice.csv`、训练日志和分割视频；点击“开始推理”会调用同一个 exe 的后台 worker 执行 `main.py` 中的分割流程。
