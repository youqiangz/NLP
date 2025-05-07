# -*- coding: utf-8 -*-

from fastapi import FastAPI
from .api.routes import router
from .config.logger import logger


app = FastAPI(
    title="NER Service API",
    version="1.0.0",
    description="Named Entity Recognition Service"
)


app.include_router(router, prefix="/api/v1", tags=["NER"])