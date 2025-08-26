"""
Test suite for etomofiles package.
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import etomofiles
from etomofiles.io import read_tlt, read_xf
from etomofiles.utils import validate_directory
from etomofiles.reader import read, _parse_edf_file, _pad_array, _pad_transform


class TestIO:
    """Test IO functions."""
    
    def test_read_tlt_success(self):
        """Test successful tlt file reading."""
        expected = np.array([-60.0, -58.0, -56.0])
        with patch("os.path.exists", return_value=True):
            with patch("numpy.loadtxt", return_value=expected):
                result = read_tlt("test.tlt")
                np.testing.assert_array_equal(result, expected)
    
    def test_read_tlt_file_not_exists(self):
        """Test tlt file reading when file doesn't exist."""
        with patch("os.path.exists", return_value=False):
            result = read_tlt("nonexistent.tlt")
            assert result is None
    
    def test_read_tlt_exception(self):
        """Test tlt file reading with exception."""
        with patch("os.path.exists", return_value=True):
            with patch("numpy.loadtxt", side_effect=Exception("Test error")):
                result = read_tlt("test.tlt")
                assert result is None
    
    def test_read_xf_success(self):
        """Test successful xf file reading."""
        expected = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                           [1.1, 0.1, 0.1, 1.1, 1.0, 2.0]])
        with patch("os.path.exists", return_value=True):
            with patch("numpy.loadtxt", return_value=expected.flatten()):
                result = read_xf("test.xf")
                np.testing.assert_array_equal(result, expected)
    
    def test_read_xf_file_not_exists(self):
        """Test xf file reading when file doesn't exist."""
        with patch("os.path.exists", return_value=False):
            result = read_xf("nonexistent.xf")
            assert result is None
    
    def test_read_xf_exception(self):
        """Test xf file reading with exception."""
        with patch("os.path.exists", return_value=True):
            with patch("numpy.loadtxt", side_effect=Exception("Test error")):
                result = read_xf("test.xf")
                assert result is None


class TestUtils:
    """Test utility functions."""
    
    def test_validate_directory_success(self):
        """Test successful directory validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validate_directory(tmpdir)  # Should not raise
    
    def test_validate_directory_not_exists(self):
        """Test directory validation when directory doesn't exist."""
        with pytest.raises(ValueError, match="Directory does not exist"):
            validate_directory("/nonexistent/path")
    
    def test_validate_directory_not_a_directory(self):
        """Test directory validation when path is not a directory."""
        with tempfile.NamedTemporaryFile() as tmpfile:
            with pytest.raises(ValueError, match="Path is not a directory"):
                validate_directory(tmpfile.name)


class TestReader:
    """Test main reader functionality."""
    
    def create_mock_edf_content(self):
        """Create mock .edf file content."""
        return """
Setup.DatasetName=TS_001
Setup.RawImageStackExt=st
Setup.ImageRotationA=83.9
Setup.AxisA.ExcludeProjections=40
"""
    
    def test_parse_edf_file_success(self):
        """Test successful .edf file parsing."""
        edf_content = self.create_mock_edf_content()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            edf_file = tmpdir / "test.edf"
            ts_file = tmpdir / "TS_001.st"
            
            # Create .edf file
            with open(edf_file, 'w') as f:
                f.write(edf_content)
            
            # Create empty tilt series file
            ts_file.touch()
            
            # Mock mrcfile
            mock_header = MagicMock()
            mock_header.nz = 40
            mock_mrc = MagicMock()
            mock_mrc.__enter__.return_value.header = mock_header
            
            with patch("mrcfile.open", return_value=mock_mrc):
                result = _parse_edf_file(tmpdir)
                
            ts_name, ts_ext, tilt_axis_angle, excluded_views, n_images = result
            
            assert ts_name == "TS_001"
            assert ts_ext == "st"
            assert tilt_axis_angle == 83.9
            assert excluded_views == {40}
            assert n_images == 40
    
    def test_parse_edf_file_no_edf(self):
        """Test .edf file parsing when no .edf file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="No .edf file found"):
                _parse_edf_file(Path(tmpdir))
    
    def test_parse_edf_file_missing_dataset_name(self):
        """Test .edf file parsing when DatasetName is missing."""
        edf_content = "Setup.RawImageStackExt=st\n"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            edf_file = tmpdir / "test.edf"
            
            with open(edf_file, 'w') as f:
                f.write(edf_content)
            
            with pytest.raises(ValueError, match="Setup.DatasetName not found"):
                _parse_edf_file(tmpdir)
    
    def test_pad_array_with_data(self):
        """Test array padding with data."""
        data = np.array([1.0, 2.0, 3.0])
        result = _pad_array(data, 5)
        expected = [1.0, 2.0, 3.0, np.nan, np.nan]
        
        assert len(result) == 5
        assert result[:3] == [1.0, 2.0, 3.0]
        assert np.isnan(result[3])
        assert np.isnan(result[4])
    
    def test_pad_array_with_none(self):
        """Test array padding with None data."""
        result = _pad_array(None, 3)
        expected = [np.nan, np.nan, np.nan]
        
        assert len(result) == 3
        assert all(np.isnan(x) for x in result)
    
    def test_pad_transform_with_data(self):
        """Test transform padding with data."""
        data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                        [1.1, 2.1, 3.1, 4.1, 5.1, 6.1]])
        result = _pad_transform(data, 4, 2)  # Column 2 (a21)
        expected = [3.0, 3.1, np.nan, np.nan]
        
        assert len(result) == 4
        assert result[:2] == [3.0, 3.1]
        assert np.isnan(result[2])
        assert np.isnan(result[3])
    
    def test_pad_transform_with_none(self):
        """Test transform padding with None data."""
        result = _pad_transform(None, 3, 0)
        
        assert len(result) == 3
        assert all(np.isnan(x) for x in result)


class TestMainRead:
    """Test main read function."""
    
    def create_test_directory(self, tmpdir):
        """Create a test directory with all necessary files."""
        tmpdir = Path(tmpdir)
        
        # Create .edf file
        edf_content = """
Setup.DatasetName=TS_001
Setup.RawImageStackExt=st
Setup.ImageRotationA=83.9
Setup.AxisA.ExcludeProjections=40
"""
        with open(tmpdir / "TS_001.edf", 'w') as f:
            f.write(edf_content)
        
        # Create tilt series file
        (tmpdir / "TS_001.st").touch()
        
        # Create alignment files
        with open(tmpdir / "TS_001.tlt", 'w') as f:
            f.write("-60.0\n-58.0\n")
        
        with open(tmpdir / "TS_001.xf", 'w') as f:
            f.write("1.0 0.0 0.0 1.0 0.0 0.0\n1.1 0.1 0.1 1.1 1.0 2.0\n")
        
        with open(tmpdir / "TS_001.rawtlt", 'w') as f:
            f.write("-60.5\n-58.2\n")
        
        with open(tmpdir / "TS_001.xtilt", 'w') as f:
            f.write("0.1\n0.2\n")
        
        return tmpdir
    
    @patch("mrcfile.open")
    @patch("etomofiles.io.read_tlt")
    @patch("etomofiles.io.read_xf")
    def test_read_success(self, mock_read_xf, mock_read_tlt, mock_mrcfile):
        """Test successful read operation."""
        # Mock mrcfile
        mock_header = MagicMock()
        mock_header.nz = 2
        mock_mrc = MagicMock()
        mock_mrc.__enter__.return_value.header = mock_header
        mock_mrcfile.return_value = mock_mrc
        
        # Mock file readers
        mock_read_tlt.side_effect = [
            np.array([-60.0, -58.0]),  # tlt
            np.array([-60.5, -58.2]),  # rawtlt
            np.array([0.1, 0.2])       # xtilt
        ]
        mock_read_xf.return_value = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                             [1.1, 0.1, 0.1, 1.1, 1.0, 2.0]])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = self.create_test_directory(tmpdir)
            
            result = read(test_dir)
            
            # Check DataFrame structure
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            
            expected_columns = [
                'image_path', 'idx_tilt', 'tilt_axis_angle', 'rawtlt', 'tlt', 
                'xtilt', 'a11', 'a12', 'a21', 'a22', 'dx', 'dy', 'excluded'
            ]
            assert list(result.columns) == expected_columns
            
            # Check specific values
            assert result['image_path'].tolist() == ['TS_001.st[0]', 'TS_001.st[1]']
            assert result['idx_tilt'].tolist() == [0, 1]
            assert all(result['tilt_axis_angle'] == 83.9)
            assert result['excluded'].tolist() == [False, False]  # No exclusions for indices 0,1
    
    def test_read_invalid_directory(self):
        """Test read with invalid directory."""
        with pytest.raises(ValueError, match="Directory does not exist"):
            read("/nonexistent/path")


class TestPackageImports:
    """Test package-level imports and exports."""
    
    def test_main_imports(self):
        """Test that main functions are importable."""
        assert hasattr(etomofiles, 'read')
        assert hasattr(etomofiles, 'read_tlt')
        assert hasattr(etomofiles, 'read_xf')
    
    def test_read_function_signature(self):
        """Test that read function has correct signature."""
        import inspect
        sig = inspect.signature(etomofiles.read)
        assert 'directory' in sig.parameters
    
    def test_package_exports(self):
        """Test that __all__ contains expected exports."""
        expected_exports = ['read', 'read_tlt', 'read_xf']
        for export in expected_exports:
            assert export in etomofiles.__all__


if __name__ == "__main__":
    pytest.main([__file__])
