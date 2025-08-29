"""
Integration tests for Hyperopt with Simplified Shared Memory
"""
import multiprocessing as mp
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from freqtrade.optimize.hyperopt_integration import HyperoptIntegration
from freqtrade.optimize.shared_memory_simple import SimpleSharedMemoryManager


def worker_process(metadata_file):
    """Worker function to test shared memory access in hyperopt context"""
    try:
        # Load shared data
        data = SimpleSharedMemoryManager.load_from_metadata(metadata_file)
        
        # Verify data is accessible
        if not data:
            return False
        
        # Perform some calculations (simulate hyperopt work)
        total_rows = sum(len(df) for df in data.values())
        
        return total_rows > 0
    except Exception as e:
        print(f"Worker error: {e}")
        return False


class TestHyperoptSimpleSharedMemory:
    """Test Hyperopt integration with simplified shared memory"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return {
            'hyperopt_jobs': 2,
            'hyperopt_batch_size': 10,
            'timeframe': '5m',
            'dry_run': True
        }
    
    @pytest.fixture
    def test_data(self):
        """Create test DataFrames"""
        data = {}
        for pair in ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']:
            data[pair] = pd.DataFrame({
                'open': np.random.rand(100),
                'high': np.random.rand(100),
                'low': np.random.rand(100),
                'close': np.random.rand(100),
                'volume': np.random.rand(100)
            })
        return data
    
    def test_hyperopt_shared_memory_setup(self, config, test_data):
        """Test setting up shared memory through HyperoptIntegration"""
        # Create temporary pickle file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle.dump(test_data, f)
            pickle_file = Path(f.name)
        
        try:
            # Initialize HyperoptIntegration
            integration = HyperoptIntegration(config)
            
            # Mock resource optimizer to recommend shared memory
            with patch.object(integration.resource_optimizer, 'should_use_shared_memory') as mock_shm:
                mock_shm.return_value = True
                
                # Setup shared memory
                metadata_file = integration.setup_shared_memory(pickle_file)
                
                assert metadata_file is not None
                assert Path(metadata_file).exists()
                
                # Verify data can be loaded
                loaded_data = SimpleSharedMemoryManager.load_from_metadata(metadata_file)
                assert len(loaded_data) == 3
                assert all(pair in loaded_data for pair in ['BTC/USDT', 'ETH/USDT', 'ADA/USDT'])
                
                # Cleanup
                integration.cleanup()
        finally:
            # Clean up pickle file
            if pickle_file.exists():
                pickle_file.unlink()
    
    def test_hyperopt_parallel_workers(self, config, test_data):
        """Test parallel workers accessing shared memory in hyperopt context"""
        # Create temporary pickle file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle.dump(test_data, f)
            pickle_file = Path(f.name)
        
        try:
            # Initialize HyperoptIntegration
            integration = HyperoptIntegration(config)
            
            # Mock resource optimizer
            with patch.object(integration.resource_optimizer, 'should_use_shared_memory') as mock_shm:
                mock_shm.return_value = True
                
                # Setup shared memory
                metadata_file = integration.setup_shared_memory(pickle_file)
                
                if metadata_file:
                    # Test with multiple worker processes
                    with mp.Pool(processes=4) as pool:
                        results = pool.map(worker_process, [metadata_file] * 4)
                    
                    assert all(results), "Not all workers could access shared memory"
                    
                    # Cleanup
                    integration.cleanup()
        finally:
            # Clean up pickle file
            if pickle_file.exists():
                pickle_file.unlink()
    
    def test_hyperopt_no_shared_memory_small_data(self, config):
        """Test that shared memory is not used for small datasets"""
        # Create small test data
        small_data = {'BTC/USDT': pd.DataFrame({'close': [1, 2, 3]})}
        
        # Create temporary pickle file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle.dump(small_data, f)
            pickle_file = Path(f.name)
        
        try:
            # Initialize HyperoptIntegration
            integration = HyperoptIntegration(config)
            
            # Mock resource optimizer to NOT recommend shared memory
            with patch.object(integration.resource_optimizer, 'should_use_shared_memory') as mock_shm:
                mock_shm.return_value = False  # Small data doesn't need shared memory
                
                # Setup should return None for small data
                metadata_file = integration.setup_shared_memory(pickle_file)
                assert metadata_file is None
                
        finally:
            # Clean up pickle file
            if pickle_file.exists():
                pickle_file.unlink()
    
    def test_hyperopt_resource_optimization(self, config):
        """Test resource optimization recommendations"""
        integration = HyperoptIntegration(config)
        
        # Get optimized settings
        settings = integration.optimize_parallel_settings()
        
        assert 'n_jobs' in settings
        assert 'batch_size' in settings
        assert 'use_shared_memory' in settings
        assert 'memory_status' in settings
        
        # Verify reasonable values
        assert settings['n_jobs'] > 0
        assert settings['batch_size'] > 0
    
    def test_hyperopt_cleanup(self, config, test_data):
        """Test proper cleanup of shared memory resources"""
        # Create temporary pickle file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle.dump(test_data, f)
            pickle_file = Path(f.name)
        
        try:
            integration = HyperoptIntegration(config)
            
            with patch.object(integration.resource_optimizer, 'should_use_shared_memory') as mock_shm:
                mock_shm.return_value = True
                
                # Setup shared memory
                metadata_file = integration.setup_shared_memory(pickle_file)
                
                if metadata_file:
                    # Verify metadata file exists
                    assert Path(metadata_file).exists()
                    
                    # Cleanup
                    integration.cleanup()
                    
                    # Verify metadata file is cleaned up
                    assert not Path(metadata_file).exists()
        finally:
            # Clean up pickle file
            if pickle_file.exists():
                pickle_file.unlink()
    
    def test_concurrent_hyperopt_runs(self, config, test_data):
        """Test that multiple hyperopt runs can use different shared memory blocks"""
        # Create two pickle files with different data
        test_data2 = {
            'XRP/USDT': pd.DataFrame({
                'close': np.random.rand(50),
                'volume': np.random.rand(50)
            })
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f1:
            pickle.dump(test_data, f1)
            pickle_file1 = Path(f1.name)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f2:
            pickle.dump(test_data2, f2)
            pickle_file2 = Path(f2.name)
        
        try:
            # Initialize two HyperoptIntegration instances
            integration1 = HyperoptIntegration(config)
            integration2 = HyperoptIntegration(config)
            
            with patch.object(integration1.resource_optimizer, 'should_use_shared_memory') as mock_shm1, \
                 patch.object(integration2.resource_optimizer, 'should_use_shared_memory') as mock_shm2:
                
                mock_shm1.return_value = mock_shm2.return_value = True
                
                # Setup shared memory for both
                metadata_file1 = integration1.setup_shared_memory(pickle_file1)
                metadata_file2 = integration2.setup_shared_memory(pickle_file2)
                
                assert metadata_file1 is not None
                assert metadata_file2 is not None
                assert metadata_file1 != metadata_file2
                
                # Verify data isolation
                data1 = SimpleSharedMemoryManager.load_from_metadata(metadata_file1)
                data2 = SimpleSharedMemoryManager.load_from_metadata(metadata_file2)
                
                assert 'BTC/USDT' in data1
                assert 'XRP/USDT' in data2
                assert 'BTC/USDT' not in data2
                assert 'XRP/USDT' not in data1
                
                # Cleanup both
                integration1.cleanup()
                integration2.cleanup()
                
        finally:
            # Clean up pickle files
            if pickle_file1.exists():
                pickle_file1.unlink()
            if pickle_file2.exists():
                pickle_file2.unlink()
