import unittest
import numpy as np
import os
import tempfile
import shutil
import psutil
from textwrap import dedent
from rich.console import Console

from ..OptiTracker import Optitracker

console = Console()


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
        self.data_file = os.path.join(self.test_dir, 'test_data.csv')
        with open(self.data_file, 'w') as f:
            f.write(self.sample_data)

        # Initialize Optitracker with test parameters
        self.tracker = Optitracker(
            marker_count=3,
            sample_rate=120,
            window_size=3,
            data_dir=self.data_file,
            rescale_by=1000,
            primary_axis='z',
        )

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        # Remove temporary test directory and files
        if os.path.exists(self.test_dir):
            os.remove(self.data_file)
            shutil.rmtree(self.test_dir)

    @unittest.skip('Not implemented')
    def test_initialization(self):
        """Test proper initialization of Optitracker object."""
        self.assertEqual(self.tracker.marker_count, 3)
        self.assertEqual(self.tracker.sample_rate, 120)
        self.assertEqual(self.tracker.window_size, 3)
        self.assertEqual(self.tracker.data_dir, self.data_file)
        self.assertEqual(self.tracker.rescale_by, 1000)

    @unittest.skip('Not implemented')
    def test_invalid_marker_count(self):
        """Test initialization with invalid marker count."""
        with self.assertRaises(ValueError):
            Optitracker(marker_count=0)

    @unittest.skip('Not implemented')
    def test_invalid_rescale_factor(self):
        """Test setting invalid rescale factor."""
        with self.assertRaises(ValueError):
            Optitracker(marker_count=5, rescale_by=0)

    def test_position(self):
        """Test position calculation."""
        pos = self.tracker.position()
        self.assertIsInstance(pos, np.ndarray)
        self.assertEqual(
            pos.dtype.names, ('frame_number', 'pos_x', 'pos_y', 'pos_z')
        )

        # Test mean position calculation for first frame
        self.assertAlmostEqual(pos['pos_x'].item(), 10.0)
        self.assertAlmostEqual(pos['pos_y'].item(), 10.0)
        self.assertAlmostEqual(pos['pos_z'].item(), 10.0)

    def test_velocity(self):
        """Test velocity calculation."""

        def expected(all):
            dd = 1
            if all:
                dd = np.sqrt(3)
            return dd / (1 / self.tracker.sample_rate)

        num_frames = 3
        velocity = self.tracker.velocity(num_frames=num_frames, axis='all')
        self.assertIsInstance(velocity, float)
        self.assertAlmostEqual(velocity, expected(True))

        velocity = self.tracker.velocity(num_frames=num_frames, axis='z')
        self.assertIsInstance(velocity, float)
        self.assertAlmostEqual(velocity, expected(False))

        velocity = self.tracker.velocity(num_frames=num_frames)
        self.assertIsInstance(velocity, float)
        self.assertAlmostEqual(velocity, expected(False))

        num_frames = 5
        velocity = self.tracker.velocity(num_frames=num_frames, axis='all')
        self.assertIsInstance(velocity, float)
        self.assertAlmostEqual(velocity, expected(True))

        velocity = self.tracker.velocity(num_frames=num_frames, axis='z')
        self.assertIsInstance(velocity, float)
        self.assertAlmostEqual(velocity, expected(False))

        velocity = self.tracker.velocity(num_frames=num_frames)
        self.assertIsInstance(velocity, float)
        self.assertAlmostEqual(velocity, expected(False))

        # Test invalid frame count
        with self.assertRaises(ValueError):
            self.tracker.velocity(num_frames=1)

    def test_distance(self):
        """Test distance calculation."""

        # get expected linear distance
        def expected(num_frames, all):
            if not all:
                return num_frames - 1
            return np.sqrt(3) * (num_frames - 1)

        num_frames = 2
        distance = self.tracker.distance(num_frames=num_frames, axis='all')
        self.assertIsInstance(distance, float)
        self.assertAlmostEqual(distance, expected(num_frames, True))

        num_frames = 2
        distance = self.tracker.distance(num_frames=num_frames, axis='z')
        self.assertIsInstance(distance, float)
        self.assertAlmostEqual(distance, expected(num_frames, False))

        num_frames = 2
        distance = self.tracker.distance(num_frames=num_frames)
        self.assertIsInstance(distance, float)
        self.assertAlmostEqual(distance, expected(num_frames, False))

        num_frames = 3
        distance = self.tracker.distance(num_frames=num_frames, axis='all')
        self.assertIsInstance(distance, float)
        self.assertAlmostEqual(distance, expected(num_frames, True))

        num_frames = 5
        distance = self.tracker.distance(num_frames=num_frames, axis='all')
        self.assertIsInstance(distance, float)
        self.assertAlmostEqual(distance, expected(num_frames, True))

        num_frames = 5
        distance = self.tracker.distance(num_frames=num_frames, axis='z')
        self.assertIsInstance(distance, float)
        self.assertAlmostEqual(distance, expected(num_frames, False))

        num_frames = 5
        distance = self.tracker.distance(num_frames=num_frames)
        self.assertIsInstance(distance, float)
        self.assertAlmostEqual(distance, expected(num_frames, False))

    @unittest.skip('Not implemented')
    def test_query_frames_validation(self):
        """Test frame querying validation."""
        # Test invalid data directory
        invalid_tracker = Optitracker(
            marker_count=2, data_dir='nonexistent.csv'
        )
        with self.assertRaises(FileNotFoundError):
            invalid_tracker.position()

        # Test negative frame count
        with self.assertRaises(ValueError):
            self.tracker.velocity(num_frames=-1)

    @unittest.skip('Not implemented')
    def test_data_rescaling(self):
        """Test position data rescaling."""
        pos = self.tracker.position()
        # Original data was in meters, should be converted to millimeters
        self.assertGreater(abs(pos['pos_x'][0]), 1)  # Should be > 1mm

    @unittest.skip('Not implemented')
    def test_empty_data_dir(self):
        """Test handling of empty data directory."""
        tracker = Optitracker(marker_count=2)
        with self.assertRaises(ValueError):
            tracker.position()

    @unittest.skip('Not implemented')
    def test_invalid_data_format(self):
        """Test handling of invalid data format."""
        # Create file with invalid format
        invalid_data = 'frame,x,y,z\n1,0.1,0.2,0.3\n'
        invalid_file = os.path.join(self.test_dir, 'invalid_data.csv')
        with open(invalid_file, 'w') as f:
            f.write(invalid_data)

        tracker = Optitracker(marker_count=2, data_dir=invalid_file)
        with self.assertRaises(ValueError):
            tracker.position()

    @unittest.skip('Not implemented')
    def test_invalid_window_size(self):
        """Test initialization with invalid window size."""
        with self.assertRaises(ValueError):
            Optitracker(marker_count=3, window_size=0)
        with self.assertRaises(ValueError):
            Optitracker(marker_count=3, window_size=-1)

    @unittest.skip('Not implemented')
    def test_invalid_sample_rate(self):
        """Test initialization with invalid sample rate."""
        with self.assertRaises(ValueError):
            Optitracker(marker_count=3, sample_rate=0)
        with self.assertRaises(ValueError):
            Optitracker(marker_count=3, sample_rate=-120)

    @unittest.skip('Not implemented')
    def test_missing_data(self):
        """Test handling of missing data in CSV."""
        data_with_gaps = dedent(
            """
            frame_number,pos_x,pos_y,pos_z
            1,0.001,0.001,0.001
            1,0.002,,0.002
            1,0.003,0.003,0.003
            2,0.004,0.004,
            2,0.005,0.005,0.005
        """
        ).strip()

        gap_file = os.path.join(self.test_dir, 'gap_data.csv')
        with open(gap_file, 'w') as f:
            f.write(data_with_gaps)

        tracker = Optitracker(marker_count=3, data_dir=gap_file)
        with self.assertRaises(ValueError):
            tracker.position()

    @unittest.skip('Not implemented')
    def test_frame_continuity(self):
        """Test handling of non-continuous frame numbers."""
        discontinuous_data = dedent(
            """
            frame_number,pos_x,pos_y,pos_z
            1,0.001,0.001,0.001
            1,0.002,0.002,0.002
            3,0.003,0.003,0.003
            3,0.004,0.004,0.004
        """
        ).strip()

        disc_file = os.path.join(self.test_dir, 'discontinuous_data.csv')
        with open(disc_file, 'w') as f:
            f.write(discontinuous_data)

        tracker = Optitracker(marker_count=2, data_dir=disc_file)
        with self.assertRaises(ValueError):
            tracker.velocity(num_frames=3)

    @unittest.skip('Not implemented')
    def test_non_numeric_data(self):
        """Test handling of non-numeric data in position columns."""
        invalid_data = dedent(
            """
            frame_number,pos_x,pos_y,pos_z
            1,0.001,invalid,0.001
            1,0.002,0.002,0.002
        """
        ).strip()

        invalid_file = os.path.join(self.test_dir, 'invalid_numeric.csv')
        with open(invalid_file, 'w') as f:
            f.write(invalid_data)

        tracker = Optitracker(marker_count=2, data_dir=invalid_file)
        with self.assertRaises(ValueError):
            tracker.position()

    @unittest.skip('Not implemented')
    def test_smoothing(self):
        """Test the smoothing functionality."""
        # Create data with noise
        noisy_data = dedent(
            """
            frame_number,pos_x,pos_y,pos_z
            1,0.001,0.001,0.001
            1,0.010,0.010,0.010
            1,-0.005,-0.005,-0.005
            2,0.002,0.002,0.002
            2,0.012,0.012,0.012
            2,-0.004,-0.004,-0.004
        """
        ).strip()

        noisy_file = os.path.join(self.test_dir, 'noisy_data.csv')
        with open(noisy_file, 'w') as f:
            f.write(noisy_data)

        tracker = Optitracker(marker_count=3, data_dir=noisy_file)
        raw_pos = tracker.position()

        # Test that smoothed data has less variance than raw data
        raw_variance = np.var(raw_pos['pos_x'])
        smooth_variance = np.var(tracker._Optitracker__smooth(frames=raw_pos))  # type: ignore
        self.assertLess(smooth_variance, raw_variance)  # type: ignore

    def test_large_dataset(self):
        """Test handling of large datasets."""
        # Create a large dataset
        num_frames = 1000
        num_markers = 10
        large_data = ['frame_number,pos_x,pos_y,pos_z']

        for frame in range(num_frames):
            for _ in range(num_markers):
                pos = f'{frame},{frame/1000:.3f},{frame/1000:.3f},{frame/1000:.3f}'
                large_data.append(pos)

        large_file = os.path.join(self.test_dir, 'large_data.csv')
        with open(large_file, 'w') as f:
            f.write('\n'.join(large_data))

        tracker = Optitracker(marker_count=num_markers, data_dir=large_file)

        # Test memory usage stays reasonable
        process = psutil.Process()
        initial_memory = process.memory_info().rss

        _ = tracker.position()
        _ = tracker.velocity(num_frames=10)
        _ = tracker.distance(num_frames=10)

        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB

        self.assertLess(
            memory_increase, 100
        )  # Should use less than 100MB additional memory


if __name__ == '__main__':
    unittest.main()
