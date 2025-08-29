"""
Test simplified shared memory implementation
"""
import multiprocessing as mp
import numpy as np
import pandas as pd
import pytest

from freqtrade.optimize.shared_memory_simple import SimpleSharedMemoryManager


def worker_func(metadata_file):
    """Worker function to test shared memory access"""
    try:
        data = SimpleSharedMemoryManager.load_from_metadata(metadata_file)
        return 'TEST/USDT' in data and len(data['TEST/USDT']) == 100
    except Exception as e:
        print(f"Worker error: {e}")
        return False


class TestSimpleSharedMemory:
    """Test the simplified shared memory implementation"""
    
    def test_basic_sharing(self):
        """Test basic DataFrame sharing and retrieval"""
        # Create test data
        test_df = pd.DataFrame({
            'open': np.random.rand(100),
            'high': np.random.rand(100),
            'low': np.random.rand(100),
            'close': np.random.rand(100),
            'volume': np.random.rand(100)
        })
        
        data_dict = {'TEST/USDT': test_df}
        
        # Share data
        smm = SimpleSharedMemoryManager()
        metadata_file = smm.share_dataframe_dict(data_dict)
        
        assert metadata_file is not None
        
        # Load in same process
        loaded_data = SimpleSharedMemoryManager.load_from_metadata(metadata_file)
        assert 'TEST/USDT' in loaded_data
        assert len(loaded_data['TEST/USDT']) == 100
        
        # Cleanup
        smm.cleanup()
    
    def test_multiprocess_access(self):
        """Test that multiple processes can access shared memory"""
        # Create test data
        test_df = pd.DataFrame({
            'open': np.random.rand(100),
            'high': np.random.rand(100),
            'low': np.random.rand(100),
            'close': np.random.rand(100),
            'volume': np.random.rand(100)
        })
        
        data_dict = {'TEST/USDT': test_df}
        
        # Share data
        smm = SimpleSharedMemoryManager()
        metadata_file = smm.share_dataframe_dict(data_dict)
        
        if metadata_file:
            # Test with multiple processes
            with mp.Pool(processes=2) as pool:
                results = pool.map(worker_func, [metadata_file] * 2)
            
            assert all(results), "Not all workers could access shared memory"
            
            # Cleanup
            smm.cleanup()
    
    def test_multiple_dataframes(self):
        """Test sharing multiple DataFrames"""
        # Create test data
        data_dict = {}
        for pair in ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']:
            data_dict[pair] = pd.DataFrame({
                'open': np.random.rand(50),
                'high': np.random.rand(50),
                'low': np.random.rand(50),
                'close': np.random.rand(50),
                'volume': np.random.rand(50)
            })
        
        # Share data
        smm = SimpleSharedMemoryManager()
        metadata_file = smm.share_dataframe_dict(data_dict)
        
        assert metadata_file is not None
        
        # Load and verify
        loaded_data = SimpleSharedMemoryManager.load_from_metadata(metadata_file)
        assert len(loaded_data) == 3
        assert all(pair in loaded_data for pair in ['BTC/USDT', 'ETH/USDT', 'ADA/USDT'])
        assert all(len(df) == 50 for df in loaded_data.values())
        
        # Cleanup
        smm.cleanup()
