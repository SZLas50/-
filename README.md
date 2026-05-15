# 信贷违约预测系统

## 项目简介

本项目是一个用于信贷违约风险预测的端到端示例系统。包含数据生成/加载、特征预处理、模型训练、模型评估与模型持久化，适合作为风控模型原型或教学演示工程。

## 功能特点

- 一键生成合成信贷数据（或加载本地数据）。
- 支持数据预处理（缺失值处理、类别编码）。
- 同时训练随机森林和逻辑回归模型，并比较两者性能。
- 输出常用评估指标（准确率、AUC、分类报告）。
- 将最优模型持久化为 joblib 文件，支持离线加载预测。

## 环境安装

推荐使用虚拟环境并通过 `requirements.txt` 安装依赖。

1. 创建并激活虚拟环境（Windows）：

```powershell
python -m venv venv
venv\Scripts\activate
```

2. 安装依赖：

```powershell
pip install -r requirements.txt
```

建议使用虚拟环境（如 `venv`）隔离依赖，避免不同项目间库版本冲突。

必要依赖（示例）：

- pandas
- numpy
- scikit-learn
- joblib
- streamlit (若使用演示页)

## 如何运行

训练并保存最优模型（若希望生成合成数据）：

```powershell
python src\main.py --train --generate-data
```

仅用已有数据训练：

```powershell
python src\main.py --train --data-path data/credit_default.csv
```

评估已保存模型：

```powershell
python src\main.py --evaluate --model-path model/joblib/model.joblib
```

运行 Streamlit 演示页（若包含 `app.py`）：

```powershell
streamlit run app.py
```

## 项目结构

- README.md — 项目说明（本文件）
- requirements.txt — Python 依赖列表
- data/ — 示例数据或用户数据（默认 data/credit_default.csv）
- model/joblib/ — 保存的模型（建议忽略上传大文件）
- src/main.py — 程序入口，包含训练/评估流程
- src/data.py — 数据生成与预处理逻辑
- src/model.py — 模型训练、评估与持久化
- app.py — Streamlit 演示应用（可选）

## 注意事项（关于 Git 推送）

生产环境下请避免将大文件或环境文件（如模型二进制、虚拟环境、临时缓存）提交到 Git。建议在仓库根添加 `.gitignore`，示例请参考 `.gitignore` 文件。

## 联系与许可

如需进一步改进（特征工程、模型调参、部署到云服务），欢迎发起 issue 或联系作者。
