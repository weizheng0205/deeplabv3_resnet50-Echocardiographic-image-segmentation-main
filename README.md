# DeepLabV3-ResNet50 超声心动图像分割

本项目基于 PyTorch 与 `torchvision.models.segmentation.deeplabv3_resnet50`，面向 EchoNet-Dynamic 超声心动图视频数据，实现左心室区域分割、Dice 指标评估、分割视频导出和桌面端可视化查看。

## 功能特点

- 使用 DeepLabV3-ResNet50 对 EchoNet-Dynamic 视频帧进行左心室分割
- 支持训练、验证、测试和从已有权重直接推理
- 输出 Dice、体积曲线、分割结果视频、CSV 指标和 PDF 图表
- 提供 Tkinter 图形界面，便于选择数据、权重、输出目录并查看结果
- 提供 PyInstaller 打包脚本，可构建 Windows 可执行程序

## 项目结构

```text
.
├── app.py              # 图形化工作台入口
├── main.py             # 训练、验证、测试与推理主程序
├── requirements.txt    # Python 依赖
├── utils/              # 数据集读取、评估指标、视频导出等工具函数
├── build_exe.ps1       # Windows PowerShell 打包脚本
├── build_exe.bat       # Windows 批处理打包脚本
├── BUILD_EXE.md        # 打包说明
└── img.png             # README 示例图片
```

> 注意：`data/`、`output/`、模型权重和生成视频体积较大，不建议直接提交到 GitHub 普通仓库。请按下面的目录约定在本地放置。

## 环境要求

- Python 3.9+
- Windows / Linux / macOS
- 推荐使用 NVIDIA GPU 与 CUDA 版本的 PyTorch

安装依赖：

```bash
pip install -r requirements.txt
```

如果需要 GPU 加速，请根据本机 CUDA 版本从 PyTorch 官网安装对应版本的 `torch` 和 `torchvision`。

## 数据集准备

本项目默认使用 EchoNet-Dynamic 数据集，目录结构如下：

```text
data/
└── EchoNet-Dynamic/
    ├── FileList.csv
    ├── VolumeTracings.csv
    └── Videos/
        ├── 0X100009310A3BD7FC.avi
        └── ...
```

相关链接：

- EchoNet-Dynamic 官网：https://echonet.github.io/dynamic/
- 论文：https://www.nature.com/articles/s41586-020-2145-8

## 命令行运行

从头训练并在测试集上评估：

```bash
python main.py --device cuda --batch_size 16 --num_workers 2
```

使用已有权重进行测试与分割视频导出：

```bash
python main.py ^
  --data_dir data/EchoNet-Dynamic ^
  --weights output/segmentation/deeplabv3_resnet50_random/best.pt ^
  --output output/segmentation/deeplabv3_resnet50_random ^
  --device cuda ^
  --batch_size 4 ^
  --num_workers 0 ^
  --run_test ^
  --save_video
```

在没有 CUDA 的机器上运行：

```bash
python main.py --device cpu --batch_size 2 --num_workers 0
```

## 图形界面

启动桌面工作台：

```bash
python app.py
```

界面中可以选择：

- 权重文件：默认 `output/segmentation/deeplabv3_resnet50_random/best.pt`
- 数据目录：默认 `data/EchoNet-Dynamic`
- 输出目录：默认 `output/segmentation/deeplabv3_resnet50_random`
- 运行设备、批大小、加载线程数和是否生成分割视频

## 输出结果

默认输出目录：

```text
output/segmentation/deeplabv3_resnet50_random/
```

常见输出包括：

- `best.pt`：验证集表现最好的模型权重
- `checkpoint.pt`：训练检查点
- `log.csv`：训练与验证日志
- `test_dice.csv` / `val_dice.csv`：测试集与验证集 Dice 指标
- `size.csv`：心室面积/体积相关曲线数据
- `videos/`：带分割遮罩的结果视频
- `size/`：可视化图表 PDF

## 打包为 EXE

Windows 环境下可使用：

```powershell
.\build_exe.ps1
```

或：

```bat
build_exe.bat
```

更多说明见 `BUILD_EXE.md`。

## 说明

本项目用于超声心动图像分割研究与学习。临床使用前需要经过严格的数据验证、模型评估和合规审查，不能直接作为医疗诊断依据。
