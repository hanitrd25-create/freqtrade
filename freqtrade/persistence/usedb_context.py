from freqtrade.persistence.custom_data import CustomDataWrapper
from freqtrade.persistence.pairlock_middleware import PairLocks
from freqtrade.persistence.trade_model import Trade

def enable_database_use() -> None:
    """
    Cleanup function to restore database usage.
    """
    PairLocks.use_db = True
    PairLocks.timeframe = ""
    Trade.use_db = True
    CustomDataWrapper.use_db = True


class FtNoDBContext:
    def __init__(self, timeframe: str = ""):
        self.timeframe = timeframe

    def __exit__(self, exc_type, exc_val, exc_tb):
        enable_database_use()
