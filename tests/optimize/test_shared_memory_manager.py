"""
Tests for Shared Memory Manager
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import pickle
from multiprocessing import shared_memory

from freqtrade.optimize.shared_memory_manager import (
    SharedMemoryManager,
    SharedDataWrapper,
    estimate_data_size
)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'open': [100, 101, 102, 103],
        'high': [101, 102, 103, 104],
        'low': [99, 100, 101, 102],
        'close': [100.5, 101.5, 102.5, 103.5],
        'volume': [1000, 1100, 1200, 1300],
    })


@pytest.fixture
def sample_dict_data(sample_dataframe):
    """Create a sample dictionary of DataFrames."""
    return {
        'BTC/USDT': sample_dataframe.copy(),
        'ETH/USDT': sample_dataframe.copy() * 0.1,
        'metadata': {'version': 1, 'pairs': 2}
    }


@pytest.fixture
def manager():
    """Create a SharedMemoryManager instance."""
    mgr = SharedMemoryManager()
    yield mgr
    # Cleanup after test
    mgr.cleanup()


def test_shared_memory_manager_init(manager):
    """Test SharedMemoryManager initialization."""
    assert manager.shared_blocks == {}
    assert manager.metadata == {}


def test_generate_key(manager, sample_dataframe):
    """Test key generation for data."""
    key1 = manager._generate_key(sample_dataframe, "test")
    assert key1.startswith("test_")
    assert len(key1) > 5
    
    # Same data should generate same key
    key2 = manager._generate_key(sample_dataframe, "test")
    assert key1 == key2
    
    # Different data should generate different key
    df2 = sample_dataframe * 2
    key3 = manager._generate_key(df2, "test")
    assert key1 != key3


def test_share_dataframe(manager, sample_dataframe):
    """Test sharing a DataFrame in shared memory."""
    key = manager.share_dataframe(sample_dataframe)
    
    assert key in manager.shared_blocks
    assert key in manager.metadata
    assert manager.metadata[key]['type'] == 'dataframe'
    assert manager.metadata[key]['shape'] == sample_dataframe.values.shape
    assert manager.metadata[key]['columns'] == sample_dataframe.columns.tolist()
    
    # Sharing again should return the same key
    key2 = manager.share_dataframe(sample_dataframe)
    assert key == key2


def test_get_dataframe(manager, sample_dataframe):
    """Test retrieving a DataFrame from shared memory."""
    key = manager.share_dataframe(sample_dataframe)
    
    # Retrieve the DataFrame
    retrieved_df = manager.get_dataframe(key)
    
    # Check that the data is the same
    pd.testing.assert_frame_equal(retrieved_df, sample_dataframe)


def test_share_dict(manager, sample_dict_data):
    """Test sharing a dictionary in shared memory."""
    key = manager.share_dict(sample_dict_data)
    
    assert key in manager.metadata
    assert manager.metadata[key]['type'] == 'dict'
    assert 'shared_keys' in manager.metadata[key]
    
    # Check that DataFrames were shared
    for k, v in sample_dict_data.items():
        if isinstance(v, pd.DataFrame):
            df_key = f"{key}_{k}"
            assert df_key in manager.metadata
            assert manager.metadata[df_key]['type'] == 'dataframe'


def test_get_dict(manager, sample_dict_data):
    """Test retrieving a dictionary from shared memory."""
    key = manager.share_dict(sample_dict_data)
    
    # Retrieve the dictionary
    retrieved_dict = manager.get_dict(key)
    
    # Check that all keys are present
    assert set(retrieved_dict.keys()) == set(sample_dict_data.keys())
    
    # Check DataFrames
    for k, v in sample_dict_data.items():
        if isinstance(v, pd.DataFrame):
            pd.testing.assert_frame_equal(retrieved_dict[k], v)
        else:
            assert retrieved_dict[k] == v


def test_cleanup_specific(manager, sample_dataframe):
    """Test cleaning up specific shared memory blocks."""
    key = manager.share_dataframe(sample_dataframe)
    
    assert key in manager.shared_blocks
    assert key in manager.metadata
    
    # Clean up specific key
    manager.cleanup(key)
    
    assert key not in manager.shared_blocks
    assert key not in manager.metadata


def test_cleanup_all(manager, sample_dataframe, sample_dict_data):
    """Test cleaning up all shared memory blocks."""
    key1 = manager.share_dataframe(sample_dataframe)
    key2 = manager.share_dict(sample_dict_data)
    
    assert len(manager.shared_blocks) > 0
    assert len(manager.metadata) > 0
    
    # Clean up all
    manager.cleanup()
    
    assert len(manager.shared_blocks) == 0
    assert len(manager.metadata) == 0


def test_cleanup_dict_with_subdataframes(manager, sample_dict_data):
    """Test that cleaning up a dict also cleans up its sub-DataFrames."""
    key = manager.share_dict(sample_dict_data)
    
    # Count initial blocks
    initial_blocks = len(manager.shared_blocks)
    assert initial_blocks > 1  # Should have main dict + DataFrames
    
    # Clean up the dict
    manager.cleanup(key)
    
    # All related blocks should be cleaned
    assert len(manager.shared_blocks) == 0


def test_error_handling_nonexistent_key(manager):
    """Test error handling for non-existent keys."""
    with pytest.raises(KeyError):
        manager.get_dataframe("nonexistent_key")
    
    with pytest.raises(KeyError):
        manager.get_dict("nonexistent_key")


def test_error_handling_wrong_type(manager, sample_dataframe):
    """Test error handling when retrieving wrong data type."""
    key = manager.share_dataframe(sample_dataframe)
    
    # Try to get DataFrame as dict
    with pytest.raises(TypeError):
        manager.get_dict(key)


def test_shared_data_wrapper_dataframe(manager, sample_dataframe):
    """Test SharedDataWrapper with DataFrame."""
    key = manager.share_dataframe(sample_dataframe)
    metadata = manager.metadata[key]
    
    wrapper = SharedDataWrapper(key, metadata)
    
    # Get data through wrapper
    retrieved_df = wrapper.get_data()
    pd.testing.assert_frame_equal(retrieved_df, sample_dataframe)
    
    # Second call should return cached data
    retrieved_df2 = wrapper.get_data()
    assert retrieved_df2 is retrieved_df  # Same object


def test_estimate_data_size_dataframe(sample_dataframe):
    """Test data size estimation for DataFrame."""
    size_mb = estimate_data_size(sample_dataframe)
    assert size_mb > 0
    assert size_mb < 1  # Small test DataFrame should be less than 1MB


def test_estimate_data_size_dict(sample_dict_data):
    """Test data size estimation for dictionary."""
    size_mb = estimate_data_size(sample_dict_data)
    assert size_mb > 0
    assert size_mb < 1  # Small test data should be less than 1MB


def test_estimate_data_size_numpy():
    """Test data size estimation for numpy array."""
    arr = np.random.randn(1000, 100)
    size_mb = estimate_data_size(arr)
    expected_size = arr.nbytes / (1024 * 1024)
    assert abs(size_mb - expected_size) < 0.01


def test_concurrent_access(manager, sample_dataframe):
    """Test that multiple processes can access shared data."""
    key = manager.share_dataframe(sample_dataframe)
    
    # Simulate access from another process
    meta = manager.metadata[key]
    
    # Access the shared memory directly
    shm = shared_memory.SharedMemory(name=meta['shm_name'])
    shared_array = np.ndarray(
        meta['shape'],
        dtype=meta['dtype'],
        buffer=shm.buf
    )
    
    # Check data is accessible
    assert shared_array.shape == sample_dataframe.values.shape
    
    # Clean up
    shm.close()


def test_custom_key(manager, sample_dataframe):
    """Test sharing with custom key."""
    custom_key = "my_custom_key"
    key = manager.share_dataframe(sample_dataframe, key=custom_key)
    
    assert key == custom_key
    assert custom_key in manager.shared_blocks
    assert custom_key in manager.metadata
    
    # Retrieve with custom key
    retrieved_df = manager.get_dataframe(custom_key)
    pd.testing.assert_frame_equal(retrieved_df, sample_dataframe)
