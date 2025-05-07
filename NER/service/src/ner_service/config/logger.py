import sys

from loguru import logger

from ..config.config import settings

# 配置loguru日志
logger.remove()  # 移除默认配置
logger.add(
    settings.logging.file,
    rotation=settings.logging.max_bytes,  # 使用配置中的max_bytes
    retention=settings.logging.backup_count,  # 使用配置中的backup_count
    level=settings.logging.level,  # 使用配置中的日志级别
    format=settings.logging.format  # 使用配置中的日志格式
)
logger.add(
    sys.stdout,
    colorize=True,
    level=settings.logging.level,
    format=settings.logging.format
)
