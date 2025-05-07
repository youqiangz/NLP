# NER Service

基于FastAPI的命名实体识别服务

## 功能特性

- 支持长文本分块处理
- 可配置的模型参数
- 自动生成API文档
- 完善的日志系统

## 快速开始

1. 安装依赖
```bash
pip install -r requirements.txt
```

2.启动服务
```bash
uvicorn ner_service:app --reload
```

3.访问Api文档
http://localhost:8000/docs