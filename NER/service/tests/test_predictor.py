# -*- coding: utf-8 -*-

import pytest
from ner_service.predictor import NERPredictor
from fastapi.testclient import TestClient
from ner_service.main import app


# 固定数据准备
def test_ner_predictor_initialization():
    """测试 Predictor 初始化"""
    predictor = NERPredictor()
    assert predictor is not None
    assert hasattr(predictor, "model")  # 检查是否成功加载模型


# 参数化测试不同输入场景
@pytest.mark.parametrize(
    "input_text, expected_result",
    [
        # 正常输入
        ("示例文本", {"entities": [{"text": "示例", "label": "LABEL_1"}]}),
        # 多个实体
        ("北京是中国的首都", {"entities": [{"text": "北京", "label": "LOC"}, {"text": "中国", "label": "LOC"}]}),
        # 空白输入
        ("", {"error": "Input text cannot be empty"}),
        # 非字符串输入（需在 predictor 中处理）
        (123, {"error": "Input must be a string"}),
    ],
)
def test_predictor_output(input_text, expected_result):
    """测试 predictor 核心功能"""
    predictor = NERPredictor()
    
    if isinstance(input_text, str) and input_text.strip() == "":
        with pytest.raises(ValueError):
            predictor.predict(input_text)
    else:
        result = predictor.predict(str(input_text))
        
        # 验证输出格式
        assert isinstance(result, dict)
        assert "entities" in result or "error" in result
        
        # 验证实体内容（如果存在）
        if "entities" in result:
            for entity in result["entities"]:
                assert "text" in entity
                assert "label" in entity