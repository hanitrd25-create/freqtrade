"""Test lazy loading functionality."""
import pandas as pd
import pytest

from freqtrade.data.history.history_utils import LazyDataLoader, load_data
from tests.conftest import get_patched_exchange, log_has


def test_lazy_loader(testdatadir, caplog):
    """Test LazyDataLoader functionality."""
    pairs = ["ETH/BTC", "LTC/BTC"]  # Use pairs that exist in testdata
    timeframe = "5m"
    
    # Test lazy loading returns LazyDataLoader
    data = load_data(
        datadir=testdatadir,
        timeframe=timeframe,
        pairs=pairs,
        lazy=True
    )
    
    assert isinstance(data, LazyDataLoader)
    assert len(data) == 0  # No data loaded yet
    
    # Access one pair - should trigger loading
    eth_data = data["ETH/BTC"]
    assert isinstance(eth_data, pd.DataFrame)
    assert len(data) == 1  # Only one pair loaded
    assert "ETH/BTC" in data
    
    # Access second pair
    ltc_data = data["LTC/BTC"]
    assert isinstance(ltc_data, pd.DataFrame)
    assert len(data) == 2  # Both pairs now loaded
    
    # Test keys() returns all potential pairs
    assert set(data.keys()) == set(pairs)
    
    # Test load_all()
    all_data = data.load_all()
    assert isinstance(all_data, dict)
    assert len(all_data) == 2
    assert "ETH/BTC" in all_data
    assert "LTC/BTC" in all_data


def test_lazy_vs_eager_loading(testdatadir):
    """Test that lazy and eager loading produce same results."""
    pairs = ["ETH/BTC", "XRP/BTC", "LTC/BTC"]
    timeframe = "5m"
    
    # Load eagerly
    eager_data = load_data(
        datadir=testdatadir,
        timeframe=timeframe,
        pairs=pairs,
        lazy=False
    )
    
    # Load lazily
    lazy_data = load_data(
        datadir=testdatadir,
        timeframe=timeframe,
        pairs=pairs,
        lazy=True
    )
    
    # Force load all lazy data
    lazy_dict = lazy_data.load_all()
    
    # Should have same pairs
    assert set(eager_data.keys()) == set(lazy_dict.keys())
    
    # Data should be identical
    for pair in eager_data:
        pd.testing.assert_frame_equal(eager_data[pair], lazy_dict[pair])


def test_lazy_loader_missing_pair(testdatadir):
    """Test LazyDataLoader with missing pair."""
    pairs = ["ETH/BTC", "NONEXISTENT/BTC"]
    timeframe = "5m"
    
    data = load_data(
        datadir=testdatadir,
        timeframe=timeframe,
        pairs=pairs,
        lazy=True,
        fail_without_data=False
    )
    
    # Should load existing pair
    assert isinstance(data["ETH/BTC"], pd.DataFrame)
    
    # Should raise KeyError for nonexistent pair
    with pytest.raises(KeyError, match="No data found for NONEXISTENT/BTC"):
        data["NONEXISTENT/BTC"]
    
    # load_all should skip missing pairs
    all_data = data.load_all()
    assert "ETH/BTC" in all_data
    assert "NONEXISTENT/BTC" not in all_data
