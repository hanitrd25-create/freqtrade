import logging

from pandas import DataFrame, to_datetime

from freqtrade.configuration import TimeRange
from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS, DEFAULT_TRADES_COLUMNS
from freqtrade.data.ipc_utils import read_compressed_ipc_to_pandas, write_compressed_ipc_from_pandas
from freqtrade.enums import CandleType, TradingMode

from .idatahandler import IDataHandler


logger = logging.getLogger(__name__)


class FeatherDataHandler(IDataHandler):
    _columns = DEFAULT_DATAFRAME_COLUMNS

    def ohlcv_store(
        self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType
    ) -> None:
        """
        Store data in json format "values".
            format looks as follows:
            [[<date>,<open>,<high>,<low>,<close>]]
        :param pair: Pair - used to generate filename
        :param timeframe: Timeframe - used to generate filename
        :param data: Dataframe containing OHLCV data
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        :return: None
        """
        filename = self._pair_data_filename(self._datadir, pair, timeframe, candle_type)
        self.create_dir_if_needed(filename)

        data.reset_index(drop=True).loc[:, self._columns].to_feather(
            filename, compression_level=9, compression="lz4"
        )

    def _ohlcv_load(
        self, pair: str, timeframe: str, timerange: TimeRange | None, candle_type: CandleType
    ) -> DataFrame:
        """
        Internal method used to load data for one pair from disk.
        Implements the loading and conversion to a Pandas dataframe.
        Timerange trimming and dataframe validation happens outside of this method.
        :param pair: Pair to load data
        :param timeframe: Timeframe (e.g. "5m")
        :param timerange: Limit data to be loaded to this timerange.
                        Optionally implemented by subclasses to avoid loading
                        all data where possible.
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        :return: DataFrame with ohlcv data, or empty DataFrame
        """
        filename = self._pair_data_filename(self._datadir, pair, timeframe, candle_type=candle_type)
        if not filename.exists():
            # Fallback mode for 1M files
            filename = self._pair_data_filename(
                self._datadir, pair, timeframe, candle_type=candle_type, no_timeframe_modify=True
            )
            if not filename.exists():
                return DataFrame(columns=self._columns)
        try:
            # Use optimized compressed IPC reading method from centralized utility
            pairdata = read_compressed_ipc_to_pandas(filename)
            
            # Ensure column names match expected format
            if len(pairdata.columns) == len(self._columns):
                pairdata.columns = self._columns
            
            # Convert date column if needed (Arrow dtypes handle this efficiently)
            if "date" in pairdata.columns:
                # Check if date is already in datetime format
                import pandas as pd
                if not pd.api.types.is_datetime64_any_dtype(pairdata["date"]):
                    pairdata["date"] = to_datetime(pairdata["date"], unit="ms", utc=True)
            
            return pairdata
        except Exception as e:
            logger.exception(
                f"Error loading data from {filename}. Exception: {e}. Returning empty dataframe."
            )
            return DataFrame(columns=self._columns)

    def ohlcv_append(
        self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType
    ) -> None:
        """
        Append data to existing data structures
        :param pair: Pair
        :param timeframe: Timeframe this ohlcv data is for
        :param data: Data to append.
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        """
        raise NotImplementedError()

    def _trades_store(self, pair: str, data: DataFrame, trading_mode: TradingMode) -> None:
        """
        Store trades data (list of Dicts) to file
        :param pair: Pair - used for filename
        :param data: Dataframe containing trades
                     column sequence as in DEFAULT_TRADES_COLUMNS
        :param trading_mode: Trading mode to use (used to determine the filename)
        """
        filename = self._pair_trades_filename(self._datadir, pair, trading_mode)
        self.create_dir_if_needed(filename)
        # Use centralized IPC writing with compression
        write_compressed_ipc_from_pandas(
            data.reset_index(drop=True), filename, compression="lz4", compression_level=9
        )

    def trades_append(self, pair: str, data: DataFrame):
        """
        Append data to existing files
        :param pair: Pair - used for filename
        :param data: Dataframe containing trades
                     column sequence as in DEFAULT_TRADES_COLUMNS
        """
        raise NotImplementedError()

    def _trades_load(
        self, pair: str, trading_mode: TradingMode, timerange: TimeRange | None = None
    ) -> DataFrame:
        """
        Load a pair from file, either .json.gz or .json
        # TODO: respect timerange ...
        :param pair: Load trades for this pair
        :param trading_mode: Trading mode to use (used to determine the filename)
        :param timerange: Limit data to be loaded to this timerange.
        :return: DataFrame containing trades, empty DataFrame if no data was found
        """
        filename = self._pair_trades_filename(self._datadir, pair, trading_mode)
        if not filename.exists():
            return DataFrame(columns=DEFAULT_TRADES_COLUMNS)

        try:
            # Use optimized compressed IPC reading method from centralized utility
            pairdata = read_compressed_ipc_to_pandas(filename)
            
            # Ensure column names match if needed
            if len(pairdata.columns) == len(DEFAULT_TRADES_COLUMNS):
                pairdata.columns = DEFAULT_TRADES_COLUMNS
            
            # Add date column if timestamp exists (Arrow dtypes handle this efficiently)
            if "timestamp" in pairdata.columns:
                pairdata["date"] = to_datetime(pairdata["timestamp"], unit="ms", utc=True)
            
            return pairdata
        except Exception as e:
            logger.exception(f"Error loading trades from {filename}: {e}")
            return DataFrame(columns=DEFAULT_TRADES_COLUMNS)

    @classmethod
    def _get_file_extension(cls):
        return "feather"
