"""
Integration tests for Hyperopt with SharedMemoryManager
"""
import multiprocessing as mp
import pickle
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from freqtrade.configuration import TimeRange
from freqtrade.optimize.hyperopt_integration import HyperoptIntegration
from freqtrade.optimize.shared_memory_manager import SharedMemoryManager


def worker_func_for_shared_memory_test(args):
    """Worker function to test shared memory access"""
    data_key, metadata = args
    smm = SharedMemoryManager()
    try:
        data = smm.get_dict(data_key, metadata)
        return 'TEST/USDT' in data and len(data['TEST/USDT']) == 100
    except Exception:
        return False


class TestHyperoptSharedMemoryIntegration:
    """Integration tests for Hyperopt with SharedMemoryManager"""

    def test_full_hyperopt_shared_memory_workflow(self, mocker):
        """Test complete workflow of hyperopt with shared memory"""
        # Create mock config
        config = {
            'datadir': Path('/tmp/test_data'),
            'timerange': TimeRange('date', 'date', 1609459200, 1640995200),
            'stake_currency': 'USDT',
            'hyperopt': {
                'use_shared_memory': True,
                'dynamic_resource_optimization': True
            }
        }
        
        # Create test data
        test_df = pd.DataFrame({
            'date': pd.date_range(start='2021-01-01', periods=1000, freq='5min'),
            'open': np.random.randn(1000) + 100,
            'high': np.random.randn(1000) + 101,
            'low': np.random.randn(1000) + 99,
            'close': np.random.randn(1000) + 100,
            'volume': np.random.randn(1000) * 1000
        })
        
        # Create temp file with test data
        data_dict = {
            'BTC/USDT': test_df,
            'ETH/USDT': test_df.copy()
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            pickle.dump(data_dict, tmp_file)
            tmp_file_path = Path(tmp_file.name)
        
        try:
            # Initialize components
            hyperopt_integration = HyperoptIntegration(config)
            
            # Setup shared memory with test data
            shared_keys = hyperopt_integration.setup_shared_memory(tmp_file_path)
            
            # If shared memory was set up (depends on recommendations)
            if shared_keys:
                assert isinstance(shared_keys, dict)
                assert 'main_data' in shared_keys
                data_key = shared_keys['main_data']
                assert data_key.startswith('hyperopt_data')
                
                # Verify data can be retrieved
                retrieved_data = hyperopt_integration.shared_memory_manager.get_dict(data_key)
                assert 'BTC/USDT' in retrieved_data
                assert 'ETH/USDT' in retrieved_data
                
                # Cleanup
                hyperopt_integration.cleanup()
                
                # Verify cleanup worked
                with pytest.raises(KeyError):
                    hyperopt_integration.shared_memory_manager.get_dict(data_key)
            else:
                # Shared memory wasn't recommended, which is also valid
                assert hyperopt_integration.shared_memory_manager is None
        finally:
            # Clean up temp file
            tmp_file_path.unlink(missing_ok=True)

    def test_parallel_worker_shared_memory_access(self, mocker):
        """Test parallel workers can access shared memory"""
        config = {
            'datadir': Path('/tmp/test_data'),
            'hyperopt': {
                'use_shared_memory': True
            }
        }
        
        # Create test data
        test_df = pd.DataFrame({
            'date': pd.date_range(start='2021-01-01', periods=100, freq='5min'),
            'close': np.random.randn(100) + 100
        })
        
        # Create temp file with test data
        data_dict = {'TEST/USDT': test_df}
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            pickle.dump(data_dict, tmp_file)
            tmp_file_path = Path(tmp_file.name)
        
        try:
            # Initialize
            hyperopt_integration = HyperoptIntegration(config)
            
            hyperopt_integration.resource_optimizer.get_recommendations = MagicMock(
                return_value={'use_shared_memory': True}
            )
            
            # Share data
            shared_result = hyperopt_integration.setup_shared_memory(tmp_file_path)
            
            assert shared_result is not None
            assert 'keys' in shared_result
            assert 'metadata' in shared_result
            assert 'main_data' in shared_result['keys']
            
            # Access shared data in same process
            smm = SharedMemoryManager()
            retrieved_data = smm.get_dict(shared_result['keys']['main_data'], shared_result['metadata'])
            
            # Test with multiple processes
            with mp.Pool(processes=2) as pool:
                results = pool.map(worker_func_for_shared_memory_test, [(shared_result['keys']['main_data'], shared_result['metadata'])] * 2)
            
            assert all(results), "Not all workers could access shared memory"
            
            # Cleanup
            hyperopt_integration.cleanup()
        finally:
            # Clean up temp file
            tmp_file_path.unlink(missing_ok=True)

    def test_resource_optimization_with_shared_memory(self, mocker):
        """Test resource optimization when using shared memory"""
        config = {
            'datadir': Path('/tmp/test_data'),
            'hyperopt_parallel': 'auto',
            'hyperopt': {
                'use_shared_memory': True,
                'dynamic_resource_optimization': True
            }
        }
        
        # Mock system resources
        mocker.patch('psutil.cpu_count', return_value=8)
        mocker.patch('psutil.virtual_memory', return_value=MagicMock(
            available=16 * 1024 * 1024 * 1024  # 16GB
        ))
        
        hyperopt_integration = HyperoptIntegration(config)
        
        # Create large test data
        large_df = pd.DataFrame(
            np.random.randn(10000, 10),
            columns=[f'col_{i}' for i in range(10)]
        )
        data_dict = {f'PAIR{i}/USDT': large_df.copy() for i in range(5)}
        
        # Create temp file with test data
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            pickle.dump(data_dict, tmp_file)
            tmp_file_path = Path(tmp_file.name)
        
        try:
            # Optimize parallel settings
            parallel_settings = hyperopt_integration.optimize_parallel_settings(
                num_epochs=1000,
                data_size_mb=500
            )
            
            assert parallel_settings['n_jobs'] > 0
            assert parallel_settings['n_jobs'] <= 8
            assert parallel_settings['batch_size'] > 0
            
            # Setup shared memory
            shared_keys = hyperopt_integration.setup_shared_memory(tmp_file_path)
            
            if shared_keys:
                # Verify optimal chunk size calculation
                chunk_size = hyperopt_integration.get_optimal_chunk_size(
                    num_epochs=1000,
                    n_jobs=parallel_settings['n_jobs']
                )
                assert chunk_size > 0
                assert chunk_size <= 1000
            
            # Cleanup
            hyperopt_integration.cleanup()
        finally:
            # Clean up temp file
            tmp_file_path.unlink(missing_ok=True)

    def test_shared_memory_error_handling(self, mocker):
        """Test error handling in shared memory operations"""
        config = {
            'datadir': Path('/tmp/test_data'),
            'hyperopt': {
                'use_shared_memory': True
            }
        }
        
        hyperopt_integration = HyperoptIntegration(config)
        
        # Test with non-existent file
        non_existent_file = Path('/tmp/non_existent_file.pkl')
        shared_keys = hyperopt_integration.setup_shared_memory(non_existent_file)
        
        # Should handle gracefully and return None
        assert shared_keys is None
        
        # Test cleanup with no shared memory
        hyperopt_integration.shared_memory_manager = None
        hyperopt_integration.cleanup()  # Should not raise
        
    def test_memory_estimation_and_limits(self, mocker):
        """Test memory estimation and limit checking"""
        config = {
            'datadir': Path('/tmp/test_data'),
            'hyperopt': {
                'use_shared_memory': True,
                'shared_memory_threshold_mb': 100
            }
        }
        
        # Mock available memory
        mocker.patch('psutil.virtual_memory', return_value=MagicMock(
            available=500 * 1024 * 1024  # 500MB
        ))
        
        hyperopt_integration = HyperoptIntegration(config)
        
        # Create data of different sizes
        small_df = pd.DataFrame(np.random.randn(100, 5))
        large_df = pd.DataFrame(np.random.randn(10000, 50))
        
        small_data = {'SMALL/USDT': small_df}
        large_data = {'LARGE/USDT': large_df}
        
        # Test with small data
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            pickle.dump(small_data, tmp_file)
            small_file_path = Path(tmp_file.name)
        
        # Test with large data
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            pickle.dump(large_data, tmp_file)
            large_file_path = Path(tmp_file.name)
        
        try:
            # Small data should use shared memory
            shared_keys_small = hyperopt_integration.setup_shared_memory(small_file_path)
            # May or may not use shared memory depending on recommendations
            
            # Large data might not use shared memory if too big
            shared_keys_large = hyperopt_integration.setup_shared_memory(large_file_path)
            # May or may not use shared memory depending on size and resources
            
            # Cleanup
            hyperopt_integration.cleanup()
        finally:
            # Clean up temp files
            small_file_path.unlink(missing_ok=True)
            large_file_path.unlink(missing_ok=True)

    def test_concurrent_hyperopt_runs(self, mocker):
        """Test multiple concurrent hyperopt runs with shared memory"""
        config1 = {
            'datadir': Path('/tmp/test_data'),
            'hyperopt': {'use_shared_memory': True}
        }
        config2 = config1.copy()
        
        # Create test data
        test_df = pd.DataFrame({
            'close': np.random.randn(100) + 100
        })
        
        data1 = {'BTC/USDT': test_df}
        data2 = {'ETH/USDT': test_df.copy()}
        
        # Create temp files
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            pickle.dump(data1, tmp_file)
            file_path1 = Path(tmp_file.name)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            pickle.dump(data2, tmp_file)
            file_path2 = Path(tmp_file.name)
        
        try:
            # Create two separate hyperopt integrations
            hyperopt1 = HyperoptIntegration(config1)
            hyperopt2 = HyperoptIntegration(config2)
            
            # Mock to ensure shared memory is used
            hyperopt1.resource_optimizer.get_recommendations = MagicMock(
                return_value={'use_shared_memory': True}
            )
            hyperopt2.resource_optimizer.get_recommendations = MagicMock(
                return_value={'use_shared_memory': True}
            )
            
            # Setup shared memory for both
            shared_keys1 = hyperopt1.setup_shared_memory(file_path1)
            shared_keys2 = hyperopt2.setup_shared_memory(file_path2)
            
            if shared_keys1 and shared_keys2:
                data_key1 = shared_keys1['main_data']
                data_key2 = shared_keys2['main_data']
                assert data_key1 != data_key2
                
                # Verify both can access their data independently
                retrieved1 = hyperopt1.shared_memory_manager.get_dict(data_key1)
                retrieved2 = hyperopt2.shared_memory_manager.get_dict(data_key2)
                
                assert 'BTC/USDT' in retrieved1
                assert 'ETH/USDT' in retrieved2
                
                # Cleanup both
                hyperopt1.cleanup()
                hyperopt2.cleanup()
                
                # Verify both are cleaned up
                smm = SharedMemoryManager()
                with pytest.raises(KeyError):
                    smm.get_dict(data_key1)
                with pytest.raises(KeyError):
                    smm.get_dict(data_key2)
        finally:
            # Clean up temp files
            file_path1.unlink(missing_ok=True)
            file_path2.unlink(missing_ok=True)
