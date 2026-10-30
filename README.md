# TMG-NNet

[![Paper](https://img.shields.io/badge/Paper-Optics%20and%20Lasers%20in%20Engineering%20196%20(2026)%20109448-b31b1b)](https://www.sciencedirect.com/journal/optics-and-lasers-in-engineering)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

**Official implementation of** *Transmission-matrix-guided untrained neural network for projecting
structured light through scattering media*.

透过散射介质投影结构光，本质是求解一个高维复值逆问题：要找到一组加载在空间光调制器（SLM）上的相位，
使得光穿过散射介质后正好在目标平面重现想要的图案。常规做法依赖海量「相位-散斑」配对数据训练网络，
而本项目用**测得的传输矩阵作为物理先验**，把网络本身当作正则化器，**不需要任何训练集**——
每投影一个图案，现场优化一次即可。

---

## 目录

- [方法概览](#方法概览)
- [与监督式方法的区别](#与监督式方法的区别)
- [代码结构](#代码结构)
- [环境依赖](#环境依赖)
- [数据准备](#数据准备)
- [运行](#运行)
- [关键参数](#关键参数)
- [实验结果](#实验结果)
- [引用](#引用)
- [开源进度](#开源进度)

---

## 方法概览

```
固定随机张量 z ──► 双分支网络 N(θ) ──► 相位 φ (N×M) ──► exp(i2πφ) ──► 复振幅 E_in
                                                                        │
                                              传输矩阵 T（实验测得） ×  ──┘
                                                        │
                                                   E_out = T · E_in
                                                        │
                                              I = |E_out|²  ──► 与目标图案比对
                                                        │
                            损失 = 强度误差 + 高斯/对比度正则 + TV 正则 ──► 更新 θ
```

1. **物理前向模型**：`E_out = T · E_in`。`T` 为实验测得的传输矩阵（采样率 γ=8），
   它把 SLM 上的复振幅映射到目标平面的复场。
2. **网络即相位**：网络输入是**固定的随机张量**（不做采样、不换数据），输出是 N×M 的相位分布。
   因此网络权重的优化过程等价于对相位本身的优化，而网络结构提供了隐式正则。
3. **实虚双分支**：`TMGnet` 用两条并行的 MLP 分支分别回归复场的实部与虚部，避免直接回归复数；
   两路输出在损失里合成 `torch.complex(real, imag)`。
4. **三项损失**：
   - `new_loss`：介质后强度与目标图案的均方误差，是拟合项；
   - `guassian_loss`：对实部/虚部方差的惩罚项，抑制能量发散、提升图案对比度（权重 `lamda1`）；
   - `TVLoss`：全变分正则，压制相位噪声（权重 `lamda2`）。
5. **优化策略**：Adam，学习率 `1e-4`，每 250 个 epoch 衰减为原来的 1/10，共 1000 个 epoch。

## 与监督式方法的区别

| | 监督式网络 | TMG-NNet（本工作） |
| --- | --- | --- |
| 训练数据 | 需要大量「相位-散斑」配对样本 | **不需要**，单个目标图案即可优化 |
| 泛化到新介质 | 需重新采集数据并训练 | 重新测一次传输矩阵即可 |
| 物理一致性 | 由数据隐式约束 | 前向模型显式写进损失 |
| 计算开销 | 一次性训练，推理快 | 每个图案需一次优化（分钟级） |

## 代码结构

```
TMG-NNet/
├── TMGnet.py        # 网络定义：实部/虚部双分支 MLP
└── train_TMG2.py    # 训练脚本：数据加载、前向模型、三项损失、优化循环
```

`TMGnet(input_size, output_size)` 的两条分支结构相同：`Linear → ReLU → Linear`，
隐藏层宽度 4096，分别输出实部与虚部。

## 环境依赖

```bash
pip install torch numpy scipy h5py
```

- Python 3.8+
- PyTorch（脚本默认使用 `cuda:0`，无 GPU 时自动回落到 CPU，但速度会慢很多）
- `scipy` / `h5py`：读取实验测得的 `.mat` 传输矩阵

## 数据准备

训练脚本从 `./matrix/` 读取两个实验标定文件：

| 文件 | 变量名 | 含义 |
| --- | --- | --- |
| `matrix/slm_mat2.mat` | `slm_mat` | SLM 端加载的相位图案序列 |
| `matrix/out_mat_40.mat` | `out_mat_40` | 对应的介质后散斑强度序列 |

默认参数下 `N=3600`、`M=1600`、采样率 `γ=8`，即相位场为 `N×M`，测量条数为 `N×γ=28800`。
换自己的实验数据时，同步修改 `train_TMG2.py` 顶部的 `N`、`M`、`gamma` 与两个 `.mat` 路径即可。

> 标定数据体积较大，未随仓库发布。如需要可联系作者。

## 运行

```bash
python train_TMG2.py
```

脚本会把优化过程中的中间结果写到 `./matrix/out_mat_40.mat` 同级的输出路径（见脚本内 `out_path`），
最终得到可加载到 SLM 上的相位分布。

## 关键参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `N` | 3600 | 输出（目标平面）模式数，也是相位场的一维尺寸 |
| `M` | 1600 | 相位场的另一维尺寸 |
| `gamma` | 8 | 传输矩阵采样率，决定测量条数 `N×gamma` |
| `learning_rate` | `1e-4` | Adam 初始学习率，每 250 epoch 衰减 10× |
| `epoch` | 1000 | 优化轮数 |
| `lamda1` | 15 | 高斯/对比度正则权重 |
| `lamda2` | 1 | 全变分正则权重 |

## 实验结果

在采样率 γ=8、光学厚度 τ≈18 的漫射玻璃后方，本方法成功投影出数字、细胞与条纹结构光图案：

- 投影图案与目标的皮尔逊相关系数 **PCC > 0.9**
- 信噪比 **SNR > 10 dB**

详见论文 *Optics and Lasers in Engineering* 196 (2026) 109448。

## 引用

```bibtex
@article{liu2026tmg,
  title   = {Transmission-matrix-guided untrained neural network for projecting structured light
             through scattering media},
  author  = {Liu, Yiyi and others},
  journal = {Optics and Lasers in Engineering},
  volume  = {196},
  pages   = {109448},
  year    = {2026}
}
```

如果这个仓库对你的研究有帮助，欢迎引用并点个 star。

## 开源进度

本仓库先放出核心的网络定义与优化脚本，方便复现方法主线。
**完整代码（传输矩阵标定、仿真对比、基线方法）将陆续补充**，issue 里留言催更也欢迎。
