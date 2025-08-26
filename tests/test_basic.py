"""
Basic tests for etomofiles package.
"""

import os
import sys
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import etomofiles
from etomofiles.io import read_tlt, read_xf
from etomofiles.utils import validate_directory
from etomofiles.reader import _pad_array, _pad_transform


def test_io_functions():
    """Test IO functions with mocking."""
    print("Testing IO functions...")
    
    # Test read_tlt
    tlt_data = "-60.0\n-58.0\n-56.0\n"
    with patch("builtins.open", mock_open(read_data=tlt_data)):
        with patch("os.path.exists", return_value=True):
            result = read_tlt("test.tlt")
            expected = np.array([-60.0, -58.0, -56.0])
            assert np.array_equal(result, expected), "read_tlt failed"
    
    # Test read_tlt with non-existent file
    with patch("os.path.exists", return_value=False):
        result = read_tlt("nonexistent.tlt")
        assert result is None, "read_tlt should return None for non-existent file"
    
    # Test read_xf
    xf_data = "1.0 0.0 0.0 1.0 0.0 0.0\n1.1 0.1 0.1 1.1 1.0 2.0\n"
    with patch("builtins.open", mock_open(read_data=xf_data)):
        with patch("os.path.exists", return_value=True):
            result = read_xf("test.xf")
            expected = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                               [1.1, 0.1, 0.1, 1.1, 1.0, 2.0]])
            assert np.array_equal(result, expected), "read_xf failed"
    
    print("IO functions tests passed")


def test_utils():
    """Test utility functions."""
    print("Testing utility functions...")
    
    # Test validate_directory with valid directory
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            validate_directory(tmpdir)
            print("validate_directory with valid directory passed")
        except Exception as e:
            print(f"validate_directory failed: {e}")
            return False
    
    # Test validate_directory with invalid directory
    try:
        validate_directory("/nonexistent/path")
        print("validate_directory should have raised ValueError")
        return False
    except ValueError:
        print("validate_directory with invalid directory passed")
    
    return True


def test_helper_functions():
    """Test helper functions."""
    print("Testing helper functions...")
    
    # Test _pad_array
    data = np.array([1.0, 2.0, 3.0])
    result = _pad_array(data, 5)
    assert len(result) == 5, "pad_array length incorrect"
    assert result[:3] == [1.0, 2.0, 3.0], "pad_array values incorrect"
    assert np.isnan(result[3]) and np.isnan(result[4]), "pad_array padding incorrect"
    
    # Test _pad_array with None
    result = _pad_array(None, 3)
    assert len(result) == 3, "pad_array with None length incorrect"
    assert all(np.isnan(x) for x in result), "pad_array with None should be all NaN"
    
    # Test _pad_transform
    data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    [1.1, 2.1, 3.1, 4.1, 5.1, 6.1]])
    result = _pad_transform(data, 4, 2)  # Column 2 (a21)
    assert len(result) == 4, "pad_transform length incorrect"
    assert result[:2] == [3.0, 3.1], "pad_transform values incorrect"
    assert np.isnan(result[2]) and np.isnan(result[3]), "pad_transform padding incorrect"
    
    print("Helper functions tests passed")


def test_package_imports():
    """Test package imports."""
    print("Testing package imports...")
    
    # Test main imports
    assert hasattr(etomofiles, 'read'), "read function not available"
    assert hasattr(etomofiles, 'read_tlt'), "read_tlt function not available"
    assert hasattr(etomofiles, 'read_xf'), "read_xf function not available"
    
    # Test __all__ exports
    expected_exports = ['read', 'read_tlt', 'read_xf']
    for export in expected_exports:
        assert export in etomofiles.__all__, f"{export} not in __all__"
    
    print("Package imports tests passed")


def main():
    """Run all basic tests."""
    print("Running Basic Tests for etomofiles")
    print("=" * 50)
    
    try:
        test_io_functions()
        test_utils()
        test_helper_functions()
        test_package_imports()
        
        print()
        print("All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 