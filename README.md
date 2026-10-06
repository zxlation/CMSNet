# CMSNet

基于 **Seven-Point Checklist（7 点检查法）** 数据集的多模态皮肤病变分类项目。模型同时利用 **临床照片（clinical）** 与 **皮肤镜图像（dermoscopy）**，联合预测：

- **诊断类别**（5 类）：痣、基底细胞癌、黑色素瘤、其他、脂溢性角化
- **七点检查特征**（7 项）：色素网络、条纹、色素沉着、退行结构、点与球、蓝白幕、血管结构

---

## 项目结构

```
CMSNet/
├── main.py           # 训练入口
├── test.py           # 测试 / 推理入口
├── model.py          # 网络结构（MLCNN）
├── dataloader.py     # 数据加载与增强
├── dependency.py     # 全局配置（路径、超参数等）
├── evaluate.py       # 评估指标
├── require.sh        # 依赖安装脚本
└── release_v0/       # 数据集划分索引与元信息
```

训练与测试均通过 `dependency.py` 中的配置项控制，无需额外命令行参数。

---

## 环境依赖

建议使用 **Python 3.8+** 与 **CUDA** 环境。

### 安装

在 `CMSNet` 目录下执行：

```bash
pip install torch torchvision
bash require.sh
pip install pandas numpy tqdm matplotlib scikit-learn
```

---

## 训练

在 `CMSNet` 目录下运行：

```bash
python main.py
```

---

## 测试

```bash
python test.py
```

---
