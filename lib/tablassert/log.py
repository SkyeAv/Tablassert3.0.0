from loguru import logger
from pathlib import Path

LOGASSERT: Path = Path('./logassert')
LOGASSERT.mkdir(parents=True, exist_ok=True)

logger.remove()

logger.add(
  LOGASSERT / 'run.log',
  level='INFO',
  format='{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}',
  rotation='10 MB',
  encoding='utf-8',
)
