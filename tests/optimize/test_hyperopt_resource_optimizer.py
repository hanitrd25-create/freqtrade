"""
Tests for Hyperopt Resource Optimizer
"""
import pytest
from unittest.mock import patch, MagicMock
from freqtrade.optimize.hyperopt_resource_optimizer import (
    HyperoptResourceOptimizer,
    optimize_hyperopt_resources
)


@pytest.fixture
def mock_system_info():
    """Mock system information for consistent testing."""
    return {
        'cpu_count': 4,
        'cpu_count_logical': 8,
        'memory_total_mb': 16384,  # 16GB
        'memory_available_mb': 8192,  # 8GB available
        'memory_percent_used': 50.0,
    }


@pytest.fixture
def optimizer_with_mock(mock_system_info):
    """Create optimizer with mocked system info."""
    with patch('psutil.cpu_count') as mock_cpu:
        with patch('psutil.virtual_memory') as mock_mem:
            # Setup CPU mocks
            mock_cpu.side_effect = lambda logical: 8 if logical else 4
            
            # Setup memory mock
            mem_mock = MagicMock()
            mem_mock.total = mock_system_info['memory_total_mb'] * 1024 * 1024
            mem_mock.available = mock_system_info['memory_available_mb'] * 1024 * 1024
            mem_mock.percent = mock_system_info['memory_percent_used']
            mock_mem.return_value = mem_mock
            
            optimizer = HyperoptResourceOptimizer()
            return optimizer


def test_hyperopt_resource_optimizer_init(optimizer_with_mock):
    """Test optimizer initialization."""
    optimizer = optimizer_with_mock
    
    assert optimizer.system_info['cpu_count'] == 4
    assert optimizer.system_info['cpu_count_logical'] == 8
    assert optimizer.system_info['memory_total_mb'] == 16384
    assert optimizer.system_info['memory_available_mb'] == 8192


def test_get_optimal_workers_cpu_limited(optimizer_with_mock):
    """Test optimal workers calculation when CPU limited."""
    optimizer = optimizer_with_mock
    
    # Without data size - should use CPU count
    workers = optimizer.get_optimal_workers()
    assert workers > 0
    assert workers <= 8  # Should not exceed logical CPU count


def test_get_optimal_workers_memory_limited(optimizer_with_mock):
    """Test optimal workers calculation when memory limited."""
    optimizer = optimizer_with_mock
    
    # With large data size - should be memory limited
    workers = optimizer.get_optimal_workers(data_size_mb=2000)
    # Each worker needs 2000MB + 512MB overhead = 2512MB
    # Available: 8192MB, so max workers = 8192 / 2512 ≈ 3
    assert workers <= 3


def test_get_optimal_workers_with_config():
    """Test that config overrides calculated workers."""
    config = {'hyperopt_jobs': 2}
    
    with patch('psutil.cpu_count', return_value=8):
        with patch('psutil.virtual_memory') as mock_mem:
            mem_mock = MagicMock()
            mem_mock.total = 16384 * 1024 * 1024
            mem_mock.available = 8192 * 1024 * 1024
            mem_mock.percent = 50.0
            mock_mem.return_value = mem_mock
            
            optimizer = HyperoptResourceOptimizer(config)
            workers = optimizer.get_optimal_workers()
            assert workers == 2  # Should use config value


def test_get_optimal_batch_size(optimizer_with_mock):
    """Test optimal batch size calculation."""
    optimizer = optimizer_with_mock
    
    # Small number of epochs
    batch_size = optimizer.get_optimal_batch_size(total_epochs=50, num_workers=4)
    assert batch_size >= optimizer.MIN_BATCH_SIZE
    assert batch_size <= optimizer.MAX_BATCH_SIZE
    assert batch_size <= 50
    
    # Large number of epochs
    batch_size = optimizer.get_optimal_batch_size(total_epochs=1000, num_workers=4)
    assert batch_size >= optimizer.MIN_BATCH_SIZE
    assert batch_size <= optimizer.MAX_BATCH_SIZE


def test_get_optimal_batch_size_memory_limited():
    """Test batch size reduction with limited memory."""
    with patch('psutil.cpu_count', return_value=4):
        with patch('psutil.virtual_memory') as mock_mem:
            # Simulate low memory
            mem_mock = MagicMock()
            mem_mock.total = 4096 * 1024 * 1024  # 4GB total
            mem_mock.available = 1500 * 1024 * 1024  # 1.5GB available
            mem_mock.percent = 63.0
            mock_mem.return_value = mem_mock
            
            optimizer = HyperoptResourceOptimizer()
            batch_size = optimizer.get_optimal_batch_size(total_epochs=100, num_workers=4)
            
            # With limited memory, batch size should be reduced
            assert batch_size <= 4  # Should not exceed number of workers


def test_should_use_shared_memory(optimizer_with_mock):
    """Test shared memory decision logic."""
    optimizer = optimizer_with_mock
    
    # Small dataset - no shared memory needed
    assert optimizer.should_use_shared_memory(data_size_mb=100) is False
    
    # Large dataset - shared memory recommended
    # Threshold is 20% of 8192MB = 1638MB
    assert optimizer.should_use_shared_memory(data_size_mb=2000) is True


def test_get_memory_status(optimizer_with_mock):
    """Test memory status reporting."""
    optimizer = optimizer_with_mock
    
    with patch('psutil.virtual_memory') as mock_mem:
        with patch('psutil.swap_memory') as mock_swap:
            # Setup mocks
            mem_mock = MagicMock()
            mem_mock.percent = 75.0
            mem_mock.available = 4096 * 1024 * 1024
            mock_mem.return_value = mem_mock
            
            swap_mock = MagicMock()
            swap_mock.percent = 25.0
            mock_swap.return_value = swap_mock
            
            status = optimizer.get_memory_status()
            
            assert status['memory_used_percent'] == 75.0
            assert status['status'] == 'healthy'
            assert len(status['recommendations']) == 0
            
            # Test high memory usage
            mem_mock.percent = 85.0
            status = optimizer.get_memory_status()
            assert status['status'] == 'warning'
            assert len(status['recommendations']) > 0
            
            # Test critical memory usage
            mem_mock.percent = 95.0
            status = optimizer.get_memory_status()
            assert status['status'] == 'critical'
            assert len(status['recommendations']) > 0


def test_adjust_for_current_load(optimizer_with_mock):
    """Test worker adjustment based on system load."""
    optimizer = optimizer_with_mock
    
    with patch('psutil.cpu_percent', return_value=50.0):
        with patch('os.getloadavg', return_value=(2.0, 2.0, 2.0)):
            # Normal load - no adjustment
            adjusted = optimizer.adjust_for_current_load(initial_workers=4)
            assert adjusted == 4
    
    with patch('psutil.cpu_percent', return_value=85.0):
        with patch('os.getloadavg', return_value=(2.0, 2.0, 2.0)):
            # High CPU usage - reduce workers
            adjusted = optimizer.adjust_for_current_load(initial_workers=4)
            assert adjusted < 4
            assert adjusted >= optimizer.MIN_WORKERS


def test_get_optimization_profile(optimizer_with_mock):
    """Test complete optimization profile generation."""
    optimizer = optimizer_with_mock
    
    with patch.object(optimizer, 'adjust_for_current_load', return_value=4):
        profile = optimizer.get_optimization_profile(
            total_epochs=100,
            data_size_mb=500
        )
        
        assert 'workers' in profile
        assert 'batch_size' in profile
        assert 'use_shared_memory' in profile
        assert 'memory_status' in profile
        assert 'recommendations' in profile
        assert 'estimated_memory_per_worker_mb' in profile
        assert 'total_memory_required_mb' in profile
        
        assert profile['workers'] == 4
        assert profile['batch_size'] > 0
        assert isinstance(profile['use_shared_memory'], bool)


def test_optimize_hyperopt_resources():
    """Test convenience function."""
    with patch('psutil.cpu_count', return_value=4):
        with patch('psutil.virtual_memory') as mock_mem:
            with patch('psutil.swap_memory') as mock_swap:
                with patch('psutil.cpu_percent', return_value=50.0):
                    with patch('os.getloadavg', return_value=(2.0, 2.0, 2.0)):
                        # Setup mocks
                        mem_mock = MagicMock()
                        mem_mock.total = 16384 * 1024 * 1024
                        mem_mock.available = 8192 * 1024 * 1024
                        mem_mock.percent = 50.0
                        mock_mem.return_value = mem_mock
                        
                        swap_mock = MagicMock()
                        swap_mock.percent = 10.0
                        mock_swap.return_value = swap_mock
                        
                        config = {}
                        profile = optimize_hyperopt_resources(
                            config,
                            total_epochs=100,
                            data_size_mb=500
                        )
                        
                        assert isinstance(profile, dict)
                        assert 'workers' in profile
                        assert 'batch_size' in profile
                        assert profile['workers'] > 0
                        assert profile['batch_size'] > 0
