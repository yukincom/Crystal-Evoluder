"""
バッチ処理ユーティリティ
"""
from typing import List, Any, Callable, Iterator
import asyncio
from concurrent.futures import ThreadPoolExecutor


class BatchProcessor:
    """バッチ処理を管理するクラス"""

    def __init__(self, batch_size: int = 100, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def process_batch(self, items: List[Any], func: Callable) -> List[Any]:
        """
        アイテムをバッチで処理

        Args:
            items: 処理対象のアイテムリスト
            func: 各アイテムに適用する関数

        Returns:
            処理結果のリスト
        """
        results = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = [func(item) for item in batch]
            results.extend(batch_results)
        return results

    async def process_batch_async(self, items: List[Any], func: Callable) -> List[Any]:
        """非同期バッチ処理"""
        results = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            tasks = [asyncio.get_event_loop().run_in_executor(self.executor, func, item) for item in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        return results

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown()


def batch_process(items: List[Any], func: Callable, batch_size: int = 100) -> Iterator[List[Any]]:
    """
    アイテムをバッチに分割して処理

    Args:
        items: 処理対象のアイテムリスト
        func: 各バッチに適用する関数
        batch_size: バッチサイズ

    Yields:
        各バッチの処理結果
    """
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
def batch_items(items: List[Any], batch_size: int) -> Iterator[List[Any]]:
    """
    アイテムをバッチサイズで分割

    Args:
        items: 分割対象のアイテムリスト
        batch_size: バッチサイズ

    Yields:
        各バッチのアイテムリスト
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]
        yield func(batch)