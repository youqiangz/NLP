from collections import namedtuple
from typing import List

Entity = namedtuple("Entity", ["type", "text", "start", "end", "confidence"])


def merge_overlap_entities(entities: List[Entity]) -> List[Entity]:
    """
    合并重叠或嵌套的实体，保留置信度最高的实体
    Args:
        entities: 实体列表，每个实体包含:
            - type: 实体类型
            - text: 实体文本
            - start: 起始位置
            - end: 结束位置
            - confidence: 置信度
    Returns:
        合并后的实体列表，按起始位置排序
    """
    if not entities:
        return []

    # 按起始位置排序，相同起始位置则按置信度降序
    entities.sort(key=lambda e: (e.start, -e.confidence))
    merged: List[Entity] = []
    for entity in entities:
        if not merged:
            merged.append(entity)
            continue
        last = merged[-1]
        if entity.start >= last.end:  # 完全不重叠
            merged.append(entity)
        elif (entity.start >= last.start and entity.end <= last.end):  # 完全包含
            if entity.confidence > last.confidence:
                merged[-1] = entity
        else:  # 部分重叠
            merged.append(entity)
    return merged
