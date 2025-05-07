# -*- coding: utf-8 -*-

from typing import List, Optional
from pydantic import BaseModel, Field


class NerEntity(BaseModel):
    type: str = Field(..., example="LOC", description="实体类型")
    text: str = Field(..., example="北京市", description="实体文本")
    start: int = Field(..., example=0, description="起始位置")
    end: int = Field(..., example=3, description="结束位置")
    confidence: float = Field(..., example=0.95, description="置信度")


class NerRequest(BaseModel):
    text: str = Field(..., example="北京是中国的首都", description="输入文本")
    threshold: Optional[float] = Field(0.5, ge=0, le=1, example=0.5, description="置信度阈值")


class NerResponse(BaseModel):
    # text: str = Field(..., description="原始文本")
    entities: List[NerEntity] = Field(..., description="识别出的实体列表")
