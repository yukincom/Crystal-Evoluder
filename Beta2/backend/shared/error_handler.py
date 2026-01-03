"""
Error handling utilities for Crystal Cluster
"""

import logging
from typing import Callable, Any, Optional, Dict, List
from collections import Counter
from functools import wraps


class ErrorCollector:
    """エラーを収集・集計するクラス"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.errors: List[Dict[str, Any]] = []
        self.success_count = 0
        self.skip_count = 0
    
    def add_error(self, context: str, error: Exception, **metadata):
        """エラーを記録"""
        self.errors.append({
            'context': context,
            'error_type': type(error).__name__,
            'error_msg': str(error)[:200],  # 最初の200文字
            'metadata': metadata
        })
        
        # デバッグログに即座に出力
        self.logger.debug(
            f"Error in {context}: {type(error).__name__} - {str(error)[:100]}"
        )
    
    def add_success(self):
        """成功カウント"""
        self.success_count += 1
    
    def add_skip(self):
        """スキップカウント"""
        self.skip_count += 1
    
    def report(self, operation_name: str, threshold: float = 0.5):
        """
        エラーサマリーをログ出力
        
        Args:
            operation_name: 処理名（例: "Neo4j update"）
            threshold: エラー率の警告閾値（0.5 = 50%）
        """
        total = self.success_count + len(self.errors)
        
        # 成功サマリー
        self.logger.info(
            f"✅ {operation_name} complete: "
            f"{self.success_count} succeeded, "
            f"{self.skip_count} skipped"
        )
        
        # エラーがなければ終了
        if not self.errors:
            return
        
        # エラーサマリー
        self.logger.warning(f"⚠️  {len(self.errors)} operations failed")
        
        # エラータイプ別集計
        error_types = Counter(e['error_type'] for e in self.errors)
        self.logger.warning("Error breakdown:")
        for err_type, count in error_types.most_common():
            self.logger.warning(f"  - {err_type}: {count} occurrences")
        
        # 最初の3件を詳細表示
        if self.errors:
            self.logger.debug(f"First 3 error details:")
            for detail in self.errors[:3]:
                self.logger.debug(
                    f"  [{detail['context']}] {detail['error_type']}: "
                    f"{detail['error_msg'][:80]}"
                )
        
        # 高エラー率の警告
        if total > 0:
            error_rate = len(self.errors) / total
            if error_rate > threshold:
                self.logger.error(
                    f"🚨 High error rate: {error_rate:.1%} of operations failed!"
                )
    
    def get_summary(self) -> Dict[str, Any]:
        """集計結果を辞書で返す"""
        return {
            'updated': self.success_count, 
            'skipped': self.skip_count,
            'failed': len(self.errors),
            'error_types': dict(Counter(e['error_type'] for e in self.errors)),
            'error_details': self.errors
        }

def safe_execute(
    func: Callable,
    args: tuple = (),
    kwargs: dict = None,
    default: Any = None,
    logger: Optional[logging.Logger] = None,
    context: str = "operation"
) -> Any:
    """
    関数を安全に実行（エラー時はデフォルト値を返す）
    
    Args:
        func: 実行する関数
        args: 位置引数
        kwargs: キーワード引数
        default: エラー時の戻り値
        logger: ロガー（Noneならログ出力しない）
        context: エラーログ用のコンテキスト情報
    
    Returns:
        func の戻り値、またはエラー時は default
    """
    if kwargs is None:
        kwargs = {}
    
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if logger:
            logger.debug(
                f"Error in {context}: {type(e).__name__} - {str(e)[:100]}"
            )
        return default


def retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    リトライ付きデコレーター
    
    Args:
        max_retries: 最大リトライ回数
        delay: 初回待機時間（秒）
        backoff: 待機時間の倍率（2.0なら指数バックオフ）
        exceptions: リトライ対象の例外タプル
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            
            current_delay = delay
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        raise  # 最後のリトライでも失敗したら例外を投げる
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
            
        return wrapper
    return decorator


# ============================================================
# 使用例（docstring）
# ============================================================
"""
使い方:

1. ErrorCollector でバッチ処理のエラー管理:

    from shared.error_handler import ErrorCollector
    
    collector = ErrorCollector(self.logger)
    
    for item in items:
        try:
            process(item)
            collector.add_success()
        except Exception as e:
            collector.add_error(f"item_{item.id}", e, item_name=item.name)
    
    collector.report("Batch processing")


2. safe_execute で個別処理の安全化:

    from shared.error_handler import safe_execute
    
    result = safe_execute(
        risky_function,
        args=(arg1, arg2),
        default=0.0,
        logger=self.logger,
        context="embedding calculation"
    )


3. retry_on_error でネットワーク処理のリトライ:

    from shared.error_handler import retry_on_error
    
    @retry_on_error(max_retries=3, delay=1.0)
    def call_api():
        return requests.get(url)
"""