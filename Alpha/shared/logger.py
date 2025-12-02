"""
共通ロガー
"""
import logging
import time
from contextlib import contextmanager

class HierarchicalLogger:
    """階層的ログ出力"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.indent_level = 0
    
    @contextmanager
    def section(self, title: str):
        """セクション開始"""
        self.logger.info(f"{'  ' * self.indent_level}▶ {title}")
        self.indent_level += 1
        start = time.time()
        
        try:
            yield
        finally:
            elapsed = time.time() - start
            self.indent_level -= 1
            self.logger.info(f"{'  ' * self.indent_level}✓ {title} ({elapsed:.2f}s)")
    
    def info(self, msg: str):
        self.logger.info(f"{'  ' * self.indent_level}{msg}")


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """ロガー設定"""
    logger = logging.Logger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    console = logging.StreamHandler()
    
    class IconFormatter(logging.Formatter):
        ICONS = {
            'DEBUG': '🔍',
            'INFO': '✨',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'CRITICAL': '💥'
        }
        def format(self, record):
            icon = self.ICONS.get(record.levelname, 'ℹ️')
            record.icon = icon
            return super().format(record)
    
    console.setFormatter(IconFormatter('%(icon)s %(message)s'))
    logger.addHandler(console)
    
    file_handler = logging.FileHandler(f'{name}.log', encoding='utf-8')
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(file_handler)
    
    return logger