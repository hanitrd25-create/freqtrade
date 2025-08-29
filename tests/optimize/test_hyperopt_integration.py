"""
Tests for Hyperopt Integration Module
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import pickle
import pandas as pd
import numpy as np

from freqtrade.optimize.hyperopt_integration import HyperoptIntegration
from freqtrade.optimize.shared_memory_manager import SharedMemoryManager


@pytest.fixture
def config():
    """Create a test configuration."""
    return {
        'user_data_dir': Path('/tmp/freqtrade'),
        'hyperopt_jobs': -1,  # Auto-detect
        'strategy': 'TestStrategy',
    }


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return {
        'BTC/USDT': pd.DataFrame({
            'open': np.random.randn(1000),
            'close': np.random.randn(1000),
            'volume': np.random.randn(1000),
        }),
        'ETH/USDT': pd.DataFrame({
            'open': np.random.randn(1000),
            'close': np.random.randn(1000),
            'volume': np.random.randn(1000),
        })
    }


def test_hyperopt_integration_init(config):
    """Test HyperoptIntegration initialization."""
    integration = HyperoptIntegration(config)
    
    assert integration.config == config
    assert integration.resource_optimizer is not None
    assert integration.shared_memory_manager is None
    assert integration.shared_data_keys == {}


@patch('freqtrade.optimize.hyperopt_integration.HyperoptResourceOptimizer')
def test_optimize_parallel_settings_auto(mock_optimizer_class, config):
    """Test parallel settings optimization with auto-detection."""
    # Setup mock
    mock_optimizer = MagicMock()
    mock_optimizer.get_recommendations.return_value = {
        'num_workers': 4,
        'batch_size': 100,
        'use_shared_memory': True,
        'memory_status': 'optimal'
    }
    mock_optimizer_class.return_value = mock_optimizer
    
    integration = HyperoptIntegration(config)
    settings = integration.optimize_parallel_settings()
    
    assert settings['n_jobs'] == 4
    assert settings['batch_size'] == 100
    assert settings['use_shared_memory'] is True
    assert settings['memory_status'] == 'optimal'


@patch('freqtrade.optimize.hyperopt_integration.HyperoptResourceOptimizer')
def test_optimize_parallel_settings_configured(mock_optimizer_class, config):
    """Test parallel settings with user-configured jobs."""
    config['hyperopt_jobs'] = 8
    
    # Setup mock
    mock_optimizer = MagicMock()
    mock_optimizer.get_recommendations.return_value = {
        'num_workers': 4,
        'batch_size': 100,
        'use_shared_memory': True,
        'memory_status': 'optimal'
    }
    mock_optimizer_class.return_value = mock_optimizer
    
    integration = HyperoptIntegration(config)
    settings = integration.optimize_parallel_settings()
    
    # Should respect user configuration
    assert settings['n_jobs'] == 8
    assert settings['batch_size'] == 100


@patch('freqtrade.optimize.hyperopt_integration.estimate_data_size')
@patch('builtins.open', new_callable=mock_open)
@patch('pickle.load')
def test_setup_shared_memory_dict(mock_pickle_load, mock_file, mock_estimate_size, config, sample_data):
    """Test shared memory setup with dictionary of DataFrames."""
    # Setup mocks
    mock_pickle_load.return_value = sample_data
    mock_estimate_size.return_value = 150.0  # 150MB
    
    integration = HyperoptIntegration(config)
    
    # Mock resource optimizer recommendations
    integration.resource_optimizer.get_recommendations = MagicMock(return_value={
        'use_shared_memory': True,
        'num_workers': 4,
        'batch_size': 100,
        'memory_status': 'optimal'
    })
    
    # Setup shared memory
    data_file = Path('/tmp/data.pkl')
    with patch.object(Path, 'exists', return_value=True):
        keys = integration.setup_shared_memory(data_file)
    
    assert keys is not None
    assert 'main_data' in keys
    assert integration.shared_memory_manager is not None
    
    # Cleanup
    if integration.shared_memory_manager:
        integration.cleanup()


@patch('freqtrade.optimize.hyperopt_integration.estimate_data_size')
@patch('builtins.open', new_callable=mock_open)
@patch('pickle.load')
def test_setup_shared_memory_not_recommended(mock_pickle_load, mock_file, mock_estimate_size, config, sample_data):
    """Test shared memory setup when not recommended."""
    # Setup mocks
    mock_pickle_load.return_value = sample_data
    mock_estimate_size.return_value = 10.0  # Small data
    
    integration = HyperoptIntegration(config)
    
    # Mock resource optimizer recommendations
    integration.resource_optimizer.get_recommendations = MagicMock(return_value={
        'use_shared_memory': False,  # Not recommended
        'num_workers': 4,
        'batch_size': 100,
        'memory_status': 'optimal'
    })
    
    # Setup shared memory
    data_file = Path('/tmp/data.pkl')
    with patch.object(Path, 'exists', return_value=True):
        keys = integration.setup_shared_memory(data_file)
    
    assert keys is None
    assert integration.shared_memory_manager is None


def test_setup_shared_memory_no_file(config):
    """Test shared memory setup with non-existent file."""
    integration = HyperoptIntegration(config)
    
    data_file = Path('/tmp/nonexistent.pkl')
    with patch.object(Path, 'exists', return_value=False):
        keys = integration.setup_shared_memory(data_file)
    
    assert keys is None
    assert integration.shared_memory_manager is None


def test_get_shared_data_wrapper(config):
    """Test getting shared data wrapper."""
    integration = HyperoptIntegration(config)
    
    # No shared memory setup
    wrapper = integration.get_shared_data_wrapper()
    assert wrapper is None
    
    # With shared memory
    integration.shared_memory_manager = MagicMock()
    integration.shared_data_keys = {'main_data': 'key1'}
    integration.shared_memory_manager.metadata = {
        'key1': {'type': 'dict', 'shape': (100, 5)}
    }
    
    wrapper = integration.get_shared_data_wrapper()
    assert wrapper is not None
    assert 'keys' in wrapper
    assert 'metadata' in wrapper
    assert wrapper['keys'] == {'main_data': 'key1'}


def test_cleanup(config):
    """Test cleanup of shared memory."""
    integration = HyperoptIntegration(config)
    
    # Setup mock shared memory
    mock_manager = MagicMock()
    integration.shared_memory_manager = mock_manager
    integration.shared_data_keys = {'test': 'key'}
    
    integration.cleanup()
    
    mock_manager.cleanup.assert_called_once()
    assert integration.shared_memory_manager is None
    assert integration.shared_data_keys == {}


@patch('freqtrade.optimize.hyperopt_integration.HyperoptResourceOptimizer')
def test_log_resource_status(mock_optimizer_class, config, capsys):
    """Test resource status logging."""
    # Setup mock
    mock_optimizer = MagicMock()
    mock_optimizer.get_resource_status.return_value = {
        'cpu_cores': 4,
        'logical_cores': 8,
        'memory_total_gb': 16.0,
        'memory_available_gb': 8.0,
        'memory_percent': 50.0,
        'cpu_percent': 25.0,
        'load_average': [1.5, 1.2, 1.0]
    }
    mock_optimizer_class.return_value = mock_optimizer
    
    integration = HyperoptIntegration(config)
    integration.log_resource_status()
    
    # Check that status was retrieved
    mock_optimizer.get_resource_status.assert_called_once()


def test_should_use_vectorized_backtest(config):
    """Test vectorized backtest decision logic."""
    integration = HyperoptIntegration(config)
    
    # Mock resource status
    integration.resource_optimizer.get_resource_status = MagicMock(return_value={
        'memory_available_gb': 8.0
    })
    
    # Compatible strategy
    strategy_config = {
        'position_stacking': False,
        'use_custom_stoploss': False,
        'trailing_stop': False
    }
    assert integration.should_use_vectorized_backtest(strategy_config) is True
    
    # Incompatible: position stacking
    strategy_config['position_stacking'] = True
    assert integration.should_use_vectorized_backtest(strategy_config) is False
    
    # Incompatible: custom stoploss
    strategy_config = {
        'position_stacking': False,
        'use_custom_stoploss': True,
        'trailing_stop': False
    }
    assert integration.should_use_vectorized_backtest(strategy_config) is False
    
    # Low memory
    integration.resource_optimizer.get_resource_status = MagicMock(return_value={
        'memory_available_gb': 1.5
    })
    strategy_config = {
        'position_stacking': False,
        'use_custom_stoploss': False,
        'trailing_stop': False
    }
    assert integration.should_use_vectorized_backtest(strategy_config) is False


def test_get_optimal_chunk_size(config):
    """Test optimal chunk size calculation."""
    integration = HyperoptIntegration(config)
    
    # Mock resource status
    integration.resource_optimizer.get_resource_status = MagicMock(return_value={
        'memory_available_gb': 8.0
    })
    
    # Small data - fits in memory
    chunk_size = integration.get_optimal_chunk_size(500.0)
    assert chunk_size == -1  # Process all at once
    
    # Large data - needs chunking
    chunk_size = integration.get_optimal_chunk_size(5000.0)
    assert chunk_size > 1  # Multiple chunks needed
