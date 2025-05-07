from typing import List
from collections import namedtuple

Chunk = namedtuple("Chunk", ["text", "start", "end"])


class TextChunker:
    @staticmethod
    def chunk_text(text: str, window_size: int, stride: int) -> List[Chunk]:
        """
        将文本分割成固定大小的块
        Args:
            text: 输入文本
            window_size: 窗口大小
            stride: 滑动步长
        Returns:
            块列表，每个块包含文本内容和起止位置
        """
        if not isinstance(text, str):
            raise TypeError("Input text must be string type")
        if window_size <= 0 or stride <= 0:
            raise ValueError("Window size and stride must be positive integers")
        chunks = []
        text_length = len(text)
        for i in range(0, text_length, stride):
            end = min(i + window_size, text_length)
            chunks.append(Chunk(text=text[i:end], start=i, end=end))
        return chunks
