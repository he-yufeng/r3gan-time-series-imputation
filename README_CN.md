# MTSIR3-GAN: 基于R3GAN的多变量时间序列插补

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

中文文档 | [English](README.md)

## 📖 项目简介

MTSIR3-GAN 是一个将 **R3GAN**（NeurIPS 2024）从图像生成迁移到**多变量时间序列插补（MTSI）**的研究项目。本仓库包含原始毕设实现以及扩展实证研究（FMGAN），系统分析了对抗性精炼在时序插补中的适用边界。

### 核心特性

- 🏗️ **现代化架构**：R3GAN-1D 一维卷积适配，Fixup 初始化，频域判别器
- 🔬 **稳健训练机制**：正则化相对论损失（RpGAN + R₁ + R₂），训练稳定无模式坍塌
- 🌐 **交互式图形界面**：基于 Dash 框架构建的 Web 界面，支持数据上传、可视化和插补
- 📊 **扩展实证研究（FMGAN）**：跨多个数据集与配置，系统研究由粗到精的对抗性精炼

> MTSIR3-GAN 与 SSGAN / TimesNet 的对比为**毕设评测结果**，汇总见 [实验结果](#-实验结果)；其来自毕设评测，可通过下方训练脚本复现。

### 扩展研究：FMGAN

[`FMGAN/`](FMGAN/) 目录包含一项关于「由粗到精的对抗性精炼何时有助于时序插补」的后续实证研究，详细成稿待发布。

## 🚀 快速开始

### 环境安装

```bash
# 克隆仓库
git clone https://github.com/he-yufeng/MTSIR3-GAN.git
cd MTSIR3-GAN

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows 系统：venv\Scripts\activate

# 安装核心依赖
pip install -r requirements.txt

# 跑 FMGAN 实证研究还需额外依赖
pip install -r FMGAN/requirements.txt
```

### 启动图形界面

```bash
cd PURE-GUIv2
python app_dad.py
```

然后在浏览器中打开 `http://127.0.0.1:8050`

## 🎮 使用方法

### 1. Web 界面（推荐新手使用）

交互式图形界面提供三个主要模块：

- **数据分析**：上传并可视化时间序列数据，分析缺失模式
- **数据插补**：使用 MTSIR3-GAN、SSGAN 或 TimesNet 进行插补
- **模型可视化**：对比模型性能并可视化结果

### 2. 从头训练模型

#### 训练 MTSIR3-GAN (R3GAN)

```bash
cd TSImputation-master/R3GAN

# 在 AirQuality 数据集上训练
python train.py --outdir=./training_runs --data=../datasets/AirQuality/pm25_missing.txt \
                --gpus=1 --batch=64 --gamma=0.5 --preset=AirQuality_MTSI

# 在 PSM 数据集上训练
python train.py --outdir=./training_runs --data=../datasets/PSM/train.csv \
                --gpus=1 --batch=32 --gamma=1.0 --preset=PSM_MTSI
```

#### 训练 SSGAN

```bash
cd TSImputation-master/SSGAN

# 编辑 main.py 设置数据集和参数
python main.py --epochs=50 --batch_size=64 --model=Based_on_BRITS
```

#### 训练 TimesNet

```bash
cd TSImputation-master/TimesNet

# 训练插补任务
python run.py --task_name imputation --data PSM --root_path ./datasets/PSM/ \
              --data_path train.csv --model_id PSM_mask_0.25 --model TimesNet \
              --mask_rate 0.25 --enc_in 25 --dec_in 25 --c_out 25 \
              --batch_size 16 --learning_rate 0.001 --train_epochs 10
```

### 3. 模型推理

```python
import torch
import numpy as np

# 加载训练好的模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('model_files/AirQuality_R3GAN.pth')
model.eval()

# 准备数据
incomplete_data = np.load('your_data.npy')  # 形状: (batch, channels, height, width)
mask = np.load('your_mask.npy')  # 1 表示观测值，0 表示缺失值

# 生成插补结果
with torch.no_grad():
    z = torch.randn(batch_size, noise_dim).to(device)
    condition = torch.from_numpy(incomplete_data * mask).to(device)
    imputed = model(z, condition)

# 与观测值结合
final = incomplete_data * mask + imputed.cpu().numpy() * (1 - mask)
```

## 📁 项目结构

每个顶层文件夹都有自己的 `README.md`，说明用法与出处。

```
MTSIR3-GAN/
├── R3GAN/                    # 毕设模型（R3GAN 迁移到 MTSI）；见 R3GAN/README.md
│   ├── train.py              # 训练入口
│   ├── gen_timeseries.py     # 生成 / 插补
│   ├── R3GAN/                # 网络架构（相对论损失 GAN）
│   └── training/             # 训练循环与损失
├── SSGAN/                    # SSGAN 基线（Miao et al., AAAI 2021）
├── TimesNet/                 # TimesNet 基线（vendored 自 THUML TSLib）
├── PURE-GUIv2.0/             # Dash Web 界面
├── FMGAN/                    # 扩展实证研究（由粗到精精炼）
│   ├── models/r3gan_1d.py    # R3GAN-1D 架构（一维适配）
│   ├── train_refiner.py      # 粗到精训练脚本
│   ├── foundation_model/     # MOMENT 基础模型封装
│   ├── evaluation/           # 评估指标与基线（SAITS/BRITS/CSDI）
│   ├── data/                 # 统一数据加载（3 种缺失模式）
│   └── scripts/              # 实验运行脚本
├── datasets/                 # 样例数据 + 下载说明；见 datasets/README.md
├── requirements.txt          # 核心依赖（FMGAN 研究另见 FMGAN/requirements.txt）
└── LICENSE                   # MIT（原创代码）；第三方文件夹各自保留许可证
```

## 📊 数据集

项目在三个标准数据集上进行了基准测试：

1. **PhysioNet Challenge 2012**：ICU 患者生命体征数据（41 个特征，超过 80% 缺失率）
2. **北京空气质量数据**：来自 36 个监测站的 PM2.5 数据（36 个特征，约 13% 缺失率）
3. **汇总服务器指标 (PSM)**：服务器性能指标数据（25 个特征）

### 数据集准备

下载数据集并放置在 `TSImputation-master/datasets/` 目录：

```
datasets/
├── AirQuality/
│   ├── pm25_ground.txt
│   └── pm25_missing.txt
├── PhysioNet/
│   └── [从 PhysioNet Challenge 2012 下载]
└── PSM/
    ├── train.csv
    ├── test.csv
    └── test_label.csv
```

**下载链接**：
- PhysioNet: https://physionet.org/content/challenge-2012/1.0.0/
- 空气质量: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
- PSM: 已包含在仓库中

## 🎯 模型架构

### MTSIR3-GAN 核心组件

1. **时间序列分块**：将 1D 序列（K×L）转换为 2D 分块（C×H×W）以供 CNN 处理
2. **多矩阵输入**：使用观测数据 + 掩码矩阵 + 时间滞后矩阵作为输入
3. **现代化主干网络**：ResNet 风格的生成器和判别器，包含：
   - 分组卷积和倒置瓶颈结构
   - Fixup 初始化（无归一化层）
   - 深度网络的残差连接
4. **正则化相对论损失**：RpGAN + R₁ + R₂ 确保稳定、多样化的生成

```
生成器：
  输入：噪声 (z) + 条件（观测数据、掩码、时间滞后）
  ↓
  嵌入层（如果是条件生成）
  ↓
  阶段 1：[上采样 → 残差块 × N]
  阶段 2：[上采样 → 残差块 × N]
  阶段 3：[上采样 → 残差块 × N]
  阶段 4：[上采样 → 残差块 × N]
  ↓
  聚合层（1×1 卷积）
  ↓
  输出：插补分块（K×H×W）

判别器：
  输入：数据分块（真实/生成）
  ↓
  提取层（1×1 卷积）
  ↓
  阶段 1：[残差块 × N → 下采样]
  阶段 2：[残差块 × N → 下采样]
  阶段 3：[残差块 × N → 下采样]
  阶段 4：[残差块 × N → 全局池化]
  ↓
  条件投影（如果是条件生成）
  ↓
  输出：真实度 logit
```

## 📈 实验结果

**毕设评测**——基准数据集上的插补误差（MAE ↓ 越低越好）。以下数字来自原毕设研究，可重跑训练脚本复现。

| 数据集       | TimesNet  | SSGAN     | MTSIR3-GAN |
|-------------|-----------|-----------|------------|
| AirQuality  | **0.396** | 0.435     | 0.412      |
| PhysioNet   | 0.656     | **0.598** | 0.631      |
| PSM (12.5%) | 0.544     | 0.586     | **0.524**  |
| PSM (25%)   | **0.649** | 0.683     | 0.671      |
| PSM (50%)   | 0.782     | 0.761     | **0.737**  |

**观察**（加粗 = 该行最优）：
- MTSIR3-GAN 与强非 GAN 基线（TimesNet）和 GAN 基线（SSGAN）相比具有竞争力，在较高缺失率的 PSM 上最优。
- 没有任何单一方法在所有数据集与缺失率上全面占优。
- 得益于正则化相对论损失，训练稳定。

> 一项关于「对抗性精炼何时有效、何时无效」的后续实证研究见 [`FMGAN/`](FMGAN/)。

## 🔧 超参数调优

关键超参数及推荐范围：

| 参数               | 描述                           | 推荐范围               |
|-------------------|-------------------------------|----------------------|
| `batch_size`      | 训练批次大小                    | 32-128               |
| `WidthPerStage`   | 网络宽度（通道数）               | [512, 512, 512, 512] - [1024, 1024, 1024, 1024] |
| `BlocksPerStage`  | 每阶段的残差块数                | [2, 2, 2, 2]          |
| `learning_rate`   | 初始学习率（配合余弦衰减）        | 1e-4 到 5e-5          |
| `gamma`           | R₁/R₂ 正则化强度                | 0.05 - 1.0            |
| `NoiseDimension`  | 潜在噪声维度                    | 64-256                |

查看 `TSImputation-master/R3GAN/train.py` 获取完整配置选项。

## 🤝 贡献指南

欢迎贡献！请随时提交 Pull Request。对于重大更改：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📝 引用

如果您发现这项工作有用，请考虑引用：

```bibtex
@article{mtsir3gan2025,
  title={MTSIR3-GAN: Adapting R3GAN for Robust Multivariate Time Series Imputation},
  author={He, Yufeng},
  year={2025}
}
```

## 📚 参考文献

主要论文和资源：

- **R3GAN**: Huang, Gokaslan, Kuleshov, Tompkin. "The GAN is dead; long live the GAN! A Modern GAN Baseline" (NeurIPS 2024)
- **SSGAN**: Miao et al. "Generative Semi-supervised Learning for Multivariate Time Series Imputation" (AAAI 2021)
- **TimesNet**: Wu et al. "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis" (ICLR 2023)
- **BRITS**: Cao et al. "BRITS: Bidirectional Recurrent Imputation for Time Series" (NeurIPS 2018)

## 📧 联系方式

- **作者**：何宇峰
- **GitHub**：[@he-yufeng](https://github.com/he-yufeng)
- **项目链接**：https://github.com/he-yufeng/MTSIR3-GAN

## 📦 第三方组件与许可证

本仓库包含来自外部项目的代码。根目录 MIT 许可证**仅覆盖原创贡献**（1D/时序适配、FMGAN 研究、GUI、工具脚本）。打包的第三方代码各自保留其许可证，详见各文件夹 README：

| 文件夹 | 上游 | 许可证 |
|--------|------|--------|
| [`R3GAN/`](R3GAN/) | NVIDIA StyleGAN3 + R3GAN（NeurIPS 2024） | **NVIDIA Source Code License**（研究/非商用） |
| [`TimesNet/`](TimesNet/) | THUML Time-Series-Library | MIT |
| [`SSGAN/`](SSGAN/) | Miao et al., AAAI 2021 | 归原作者 |

> ⚠️ `R3GAN/` 含 NVIDIA 许可代码，该目录**不可**按 MIT 使用——复用须遵守上游非商用条款。

## 📄 许可证

本仓库原创代码采用 MIT 许可证 - 详见 [LICENSE](LICENSE)。打包的第三方代码见上方「第三方组件」。

## 🙏 致谢

- R3GAN 实现改编自官方仓库
- SSGAN 和 TimesNet 实现基于各自论文
- GUI 框架使用 Plotly Dash 构建
- 数据集来自 PhysioNet、UCI ML Repository 和 eBay
