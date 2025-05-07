# -*- coding: utf-8 -*-

from pathlib import Path
import sys

from httpx import AsyncClient
import pytest

from src.ner_service.main import app

# 将项目根目录添加到 Python 路径
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR / "src"))


@pytest.mark.asyncio
async def test_predict_endpoint():
    with open(BASE_DIR / "data" / "sample_input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    test_data = {
        "text": text,
        "threshold": 0.5
    }

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/api/v1/predict", json=test_data)

    assert response.status_code == 200
    response_data = response.json()
    print(response_data)

    assert "entities" in response_data
    assert isinstance(response_data["entities"], list)

    if len(response_data["entities"]) > 0:
        entity = response_data["entities"][0]
        required_keys = ["type", "text", "start", "end", "confidence"]
        assert all(key in entity for key in required_keys)
