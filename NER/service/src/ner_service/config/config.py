# -*- coding: utf-8 -*-

from pydantic import BaseModel
import yaml


class AppConfig(BaseModel):
    name: str
    version: str
    description: str
    debug: bool


class ModelConfig(BaseModel):
    path: str
    config_path: str
    tokenizer: str
    labels: str
    device: str
    head_size: int
    max_len: int
    stride: int
    batch_size: int
    threshold: float


class ServerConfig(BaseModel):
    host: str
    port: int
    workers: int


class LoggingConfig(BaseModel):
    level: str
    format: str
    file: str
    max_bytes: int
    backup_count: int


class Settings(BaseModel):
    app: AppConfig
    model: ModelConfig
    server: ServerConfig
    logging: LoggingConfig


def load_config(config_path: str = "config/settings.yaml") -> Settings:
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    return Settings(**config_data)


settings = load_config()
