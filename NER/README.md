# NER (命名实体识别) 模块

## 项目概述

本模块是NLP仓库的命名实体识别（Named Entity Recognition）实现，包含完整的训练和服务部署功能。项目采用了现代化的NLP技术栈，支持高效的实体识别和处理。

## 目录结构

```
NER/
├── README.md           # 项目说明文档
├── checkpoint/         # 模型检查点保存目录
├── service/            # 服务部署相关代码
│   ├── config/         # 服务配置文件
│   ├── data/           # 服务所需数据
│   ├── logs/           # 日志文件
│   ├── scripts/        # 部署脚本
│   ├── src/            # 服务源代码
│   ├── tests/          # 测试代码
│   ├── Jenkinsfile     # CI/CD配置
│   ├── LICENSE         # 许可证
│   ├── Makefile        # 构建配置
│   ├── README.md       # 服务说明文档
│   ├── requirements.txt # 依赖包列表
│   └── setup.py        # 安装脚本
└── train/              # 模型训练相关代码
    ├── algo/           # 算法实现
    │   └── ner_global_pointer.py # 基于GlobalPointer的NER实现
    ├── config/         # 训练配置
    │   └── config.yaml # 配置文件
    ├── data/           # 训练数据
    ├── models/         # 模型目录
    │   └── pretrained/ # 预训练模型
    └── utils/          # 工具函数
        ├── data_process.py # 数据处理
        └── utils.py    # 通用工具函数
```

## 功能模块

### 1. 训练模块 (train/)

训练模块实现了基于GlobalPointer的命名实体识别算法，主要特点：

- 采用bert4torch框架实现
- 使用GlobalPointer机制进行实体识别
- 支持多种预训练模型
- 数据处理支持长文本分块、实体过滤等功能
- 配置灵活，可通过config.yaml调整参数

### 2. 服务模块 (service/)

基于FastAPI实现的命名实体识别服务，主要特点：

- 支持长文本分块处理
- 可配置的模型参数
- 自动生成API文档
- 完善的日志系统
- 包含CI/CD配置和部署脚本

## 快速开始

### 训练模型

1. 准备数据
   - 将训练数据放入`train/data/`目录
   - 数据格式为jsonl，每行包含text和entities字段

2. 配置参数
   - 修改`train/config/config.yaml`中的配置参数

3. 开始训练
   ```bash
   # 进入训练目录
   cd train
   # 运行训练脚本
   python algo/ner_global_pointer.py
   ```

### 部署服务

1. 安装依赖
   ```bash
   cd service
   pip install -r requirements.txt
   ```

2. 启动服务
   ```bash
   uvicorn ner_service:app --reload
   ```

3. 访问API文档
   - 打开浏览器访问：http://localhost:8000/docs

## 技术细节

- 算法实现：基于GlobalPointer的命名实体识别
- 预训练模型：支持各种Transformer类预训练模型
- 数据处理：支持长文本分块、实体过滤、标签收集等功能
- 服务框架：FastAPI
- 依赖管理：requirements.txt

## 注意事项

- 训练前请确保已下载预训练模型到`train/models/pretrained/`目录
- 服务部署时需要将训练好的模型放入正确的目录
- 配置文件中的路径均为相对路径，请勿随意修改目录结构