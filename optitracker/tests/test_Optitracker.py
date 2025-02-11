import unittest
import numpy as np
import os
import tempfile
import shutil
from textwrap import dedent

from ..Optitracker import Optitracker


class TestOptitracker(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()

        self.sample_data = dedent(
            """
            frame_number,pos_x,pos_y,pos_z
            1,0.000,0.000,0.000
            1,0.001,0.001,0.001
            1,0.002,0.002,0.002
            2,0.001,0.001,0.001
            2,0.002,0.002,0.002
            2,0.003,0.003,0.003
            3,0.002,0.002,0.002
            3,0.003,0.003,0.003
            3,0.004,0.004,0.004
            4,0.003,0.003,0.003
            4,0.004,0.004,0.004
            4,0.005,0.005,0.005
            5,0.004,0.004,0.004
            5,0.005,0.005,0.005
            5,0.006,0.006,0.006
            6,0.005,0.005,0.005
            6,0.006,0.006,0.006
            6,0.007,0.007,0.007
            7,0.006,0.006,0.006
            7,0.007,0.007,0.007
            7,0.008,0.008,0.008
            8,0.007,0.007,0.007
            8,0.008,0.008,0.008
            8,0.009,0.009,0.009
            9,0.008,0.008,0.008
            9,0.009,0.009,0.009
            9,0.010,0.010,0.010
            10,0.009,0.009,0.009
            10,0.010,0.010,0.010
            10,0.011,0.011,0.011
            """
        ).strip()
        # Create sample tracking data

        # Write sample data to temporary file
        self.data_file = os.path.join(self.test_dir, "test_data.csv")
        with open(self.data_file, "w") as f:
            f.write(self.sample_data)

        # Initialize Optitracker with test parameters
        self.tracker = Optitracker(
            marker_count=3,
            sample_rate=120,
            window_size=3,
            data_dir=self.data_file,
            rescale_by=1000,
        )

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        # Remove temporary test directory and files
        if os.path.exists(self.test_dir):
            os.remove(self.data_file)
            shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test proper initialization of Optitracker object."""
        self.assertEqual(self.tracker.marker_count, 3)
        self.assertEqual(self.tracker.sample_rate, 120)
        self.assertEqual(self.tracker.window_size, 3)
        self.assertEqual(self.tracker.data_dir, self.data_file)
        self.assertEqual(self.tracker.rescale_by, 1000)

    def test_invalid_marker_count(self):
        """Test initialization with invalid marker count."""
        with self.assertRaises(ValueError):
            Optitracker(marker_count=0)

    def test_invalid_rescale_factor(self):
        """Test setting invalid rescale factor."""
        with self.assertRaises(ValueError):
           Optitracker(marker_count=5, rescale_by=0)

    def test_position(self):
        """Test position calculation."""
        pos = self.tracker.position()
        self.assertIsInstance(pos, np.ndarray)
        self.assertEqual(pos.dtype.names, ("frame_number", "pos_x", "pos_y", "pos_z"))

        # Test mean position calculation for first frame
        self.assertAlmostEqual(pos["pos_x"].item(), 10.0)        
        self.assertAlmostEqual(pos["pos_y"].item(), 10.0)        
        self.assertAlmostEqual(pos["pos_z"].item(), 10.0)        


    def test_velocity(self):
        """Test velocity calculation."""
        velocity = self.tracker.velocity(num_frames=2)
        self.assertIsInstance(velocity, float)
        self.assertGreaterEqual(velocity, 0)

        # Test invalid frame count
        with self.assertRaises(ValueError):
            self.tracker.velocity(num_frames=1)

    def test_distance(self):
        """Test distance calculation."""
        distance = self.tracker.distance(num_frames=2)
        self.assertIsInstance(distance, float)
        self.assertGreaterEqual(distance, 0)

    def test_query_frames_validation(self):
        """Test frame querying validation."""
        # Test invalid data directory
        invalid_tracker = Optitracker(marker_count=2, data_dir="nonexistent.csv")
        with self.assertRaises(FileNotFoundError):
            invalid_tracker.position()

        # Test negative frame count
        with self.assertRaises(ValueError):
            self.tracker.velocity(num_frames=-1)

    def test_data_rescaling(self):
        """Test position data rescaling."""
        pos = self.tracker.position()
        # Original data was in meters, should be converted to millimeters
        self.assertGreater(abs(pos["pos_x"][0]), 1)  # Should be > 1mm

    def test_empty_data_dir(self):
        """Test handling of empty data directory."""
        tracker = Optitracker(marker_count=2)
        with self.assertRaises(ValueError):
            tracker.position()

    def test_invalid_data_format(self):
        """Test handling of invalid data format."""
        # Create file with invalid format
        invalid_data = "frame,x,y,z\n1,0.1,0.2,0.3\n"
        invalid_file = os.path.join(self.test_dir, "invalid_data.csv")
        with open(invalid_file, "w") as f:
            f.write(invalid_data)

        tracker = Optitracker(marker_count=2, data_dir=invalid_file)
        with self.assertRaises(ValueError):
            tracker.position()


if __name__ == "__main__":
    unittest.main()
