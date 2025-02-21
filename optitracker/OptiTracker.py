import os
import numpy as np
from rich.console import Console

# import warnings

# from klibs.KLDatabase import KLDatabase as kld

# TODO:
# grab first frame, row count indicates num markers tracked.
# incorporate checks to ensure frames queried match expected marker count
# refactor nomeclature about frame indexing/querying


class Optitracker(object):
    """A class for processing and analyzing 3D motion tracking data.

    This class handles loading, processing, and analyzing positional data from motion
    tracking markers. It provides functionality for calculating velocities, positions,
    and distances in 3D space, with optional smoothing using Butterworth filtering.

    Attributes:
        marker_count (int): Number of markers being tracked
        sample_rate (int): Data sampling rate in Hz
        window_size (int): Size of the temporal window for calculations (in frames)
        data_dir (str): Path to the data file containing tracking data
        rescale_by (float): Factor to rescale position data (e.g., 1000 for m to mm); default is 1000

    Note:
        I cannot get Motive to return millimeters (vs meters) for position data, so
        position data is automatically rescaled by multiplying with rescale_by factor.
    """

    def __init__(
        self,
        marker_count: int,
        sample_rate: int = 120,
        window_size: int = 5,
        data_dir: str = '',
        rescale_by: int | float = 1000,
    ):
        """Initialize the OptiTracker object.

        Args:
            marker_count (int): Number of markers being tracked
            sample_rate (int, optional): Data sampling rate in Hz. Defaults to 120.
            window_size (int, optional): Number of frames for temporal calculations. Defaults to 5.
            data_dir (str, optional): Path to the tracking data file. Defaults to "".
            rescale_by (float, optional): Factor to rescale position values. Defaults to 1000.

        Raises:
            ValueError: If marker_count is non-positive integer
            ValueError: If sample_rate is non-positive integer
            ValueError: If window_size is non-positive integer
            ValueError: If rescale_by is non-positive numeric
        """

        self.console = Console()
        if marker_count <= 0:
            raise ValueError('Marker count must be positive.')
        self.__marker_count = marker_count

        if sample_rate <= 0:
            raise ValueError('Sample rate must be positive.')
        self.__sample_rate = sample_rate

        if window_size < 1:
            raise ValueError('Window size must be postively non-zero.')
        self.__window_size = window_size

        if rescale_by <= 0.0:
            raise ValueError('Rescale factor must be positive')
        self.__rescale_by = rescale_by

        self.__data_dir = data_dir

    @property
    def marker_count(self) -> int:
        """Get the number of markers to track."""
        return self.__marker_count

    @property
    def data_dir(self) -> str:
        """Get the data directory path."""
        return self.__data_dir

    @data_dir.setter
    def data_dir(self, data_dir: str) -> None:
        """Set the data directory path."""
        self.__data_dir = data_dir

    @property
    def sample_rate(self) -> int:
        """Get the sampling rate."""
        return self.__sample_rate

    @property
    def rescale_by(self) -> int | float:
        """Get the rescaling factor used to convert position data.

        This factor is multiplied with all position values after reading from file.
        For example, use 1000 to convert meters to millimeters.
        """
        return self.__rescale_by

    @property
    def window_size(self) -> int:
        """Get the window size."""
        return self.__window_size

    def velocity(self, num_frames: int = 0) -> float:
        """Calculate the current velocity from position data.

        Computes velocity by measuring displacement over time using the specified
        number of frames or the default window size.

        Args:
            num_frames (int, optional): Number of frames to use for calculation.
                If 0, uses the instance's window_size. Defaults to 0.

        Returns:
            float: Calculated velocity in units/second (based on rescale_by factor)

        Raises:
            ValueError: If num_frames is less than 2
        """
        if num_frames == 0:
            num_frames = self.__window_size

        if num_frames < 2:
            raise ValueError('Window size must cover at least two frames.')

        frames = self.__query_frames(num_frames)

        velocities = self.__calc_vector_velocity(frames)

        return np.mean(velocities['velocity'], dtype=np.float64)

    def position(self) -> np.ndarray:
        """Get the current mean position across all markers.

        Returns:
            np.ndarray: Structured array containing the mean position with fields:
                - frame_number (int): Frame identifier
                - pos_x (float): X coordinate
                - pos_y (float): Y coordinate
                - pos_z (float): Z coordinate
        """
        frame = self.__query_frames(num_frames=1)

        return self.__calc_position(frames=frame)

    def distance(self, num_frames: int = 0) -> float:
        """Calculate the Euclidean distance traveled over specified frames.

        Args:
            num_frames (int, optional): Number of frames to calculate distance over.
                If 0, uses the instance's window_size. Defaults to 0.

        Returns:
            float: Euclidean distance between start and end positions

        Note:
            Distance is calculated using smoothed position data if smoothing is enabled
        """

        if num_frames == 0:
            num_frames = self.__window_size

        # frames = self.__query_frames(num_frames)
        # return self.__calc_vector_distance(frames=frames)
        pass

    def __calc_vector_velocity(
        self, frames: np.ndarray = np.array([])
    ) -> np.ndarray:
        """
        Calculate velocity using position data over the specified window.

        Args:
            frames (np.ndarray, optional): Array of frame data; queries last window_size frames if empty.

        Returns:
            float: Calculated velocity in cm/s
        """
        if self.__window_size < 2:
            raise ValueError('Window size must cover at least two frames.')

        if len(frames) == 0:
            frames = self.__query_frames()

        distances = self.__calc_vector_distance(frames)
        velocities = np.ndarray(
            len(distances),
            dtype=[
                ('frame_number', 'i8'),
                ('velocity', 'f8'),
            ],
        )

        velocities['frame_number'][:] = distances['frame_number']
        velocities['velocity'][:] = distances['distance'] / (
            1.0 / self.__sample_rate
        )

        return velocities

    def __calc_vector_distance(
        self, frames: np.ndarray = np.array([])
    ) -> np.ndarray:
        """
        Calculate Euclidean distance between first and last frames.

        Args:
            frames (np.ndarray, optional): Array of frame data; queries last window_size frames if empty.

        Returns:
            float: Euclidean distance
        """
        positions = self.__calc_position(frames)
        distances = np.ndarray(
            len(positions) - 1,
            dtype=[
                ('frame_number', 'i8'),
                ('distance', 'f8'),
            ],
        )

        distances['frame_number'][:] = positions['frame_number'][1:]
        for i in range(len(positions) - 1):
            distances['frame_number'][i] = positions['frame_number'][i + 1]

            variance = np.sum(
                [
                    np.diff(positions['pos_x'][i : i + 2]) ** 2,
                    np.diff(positions['pos_y'][i : i + 2]) ** 2,
                    np.diff(positions['pos_z'][i : i + 2]) ** 2,
                ]
            )
            deviation = np.sqrt(variance)
            distances['distance'][i] = deviation

        return distances

    def __calc_position(self, frames: np.ndarray = np.array([])) -> np.ndarray:
        """Calculate mean positions across all markers for each frame.

        For each frame, computes the centroid position by averaging the positions
        of all markers tracked in that frame.

        Args:
            frames (np.ndarray, optional): Structured array of frame data.
                If empty, queries last window_size frames. Defaults to empty array.

        Returns:
            np.ndarray: Structured array of mean positions with fields:
                - frame (int): Frame identifier
                - pos_x (float): Mean X coordinate
                - pos_y (float): Mean Y coordinate
                - pos_z (float): Mean Z coordinate

        Note:
            If smooth=True, means are filtered using the __smooth method
        """
        if len(frames) == 0:
            frames = self.__query_frames()

        print('\n\n | __column_means START | \n\n')

        # Create output array with the correct dtype
        positions = np.zeros(
            len(frames) // self.__marker_count,
            dtype=[
                ('frame_number', 'i8'),
                ('pos_x', 'f8'),
                ('pos_y', 'f8'),
                ('pos_z', 'f8'),
            ],
        )

        positions['frame_number'][:] = frames['frame_number'][
            :: self.__marker_count
        ]
        for axis in ['pos_x', 'pos_y', 'pos_z']:
            positions[axis][:] = np.mean(
                frames[axis].reshape(-1, self.__marker_count), axis=1
            )

        print('Positions: ', positions)

        print('\n\n | __column_means END | \n\n')

        return positions

    # TODO: should default to None
    def __query_frames(self, num_frames: int = 0) -> np.ndarray:
        """Load and process frame data from the tracking data file.

        Reads position data from CSV file, validates format, and applies rescaling.
        Returns the most recent frames up to the specified number.

        Args:
            num_frames (int, optional): Number of most recent frames to return.
                If 0, uses the instance's window_size. Defaults to 0.

        Returns:
            np.ndarray: Structured array of frame data with fields:
                - frame_number (int): Frame identifier
                - pos_x (float): X coordinate (rescaled)
                - pos_y (float): Y coordinate (rescaled)
                - pos_z (float): Z coordinate (rescaled)

        Raises:
            ValueError: If data_dir is empty or data format is invalid
            FileNotFoundError: If data file does not exist
            ValueError: If num_frames is negative
            ValueError: If rescale_by is not positive

        Note:
            Position values are automatically multiplied by rescale_by factor
        """

        if self.__data_dir == '':
            raise ValueError('No data directory was set.')

        if not os.path.exists(self.__data_dir):
            raise FileNotFoundError(
                f'Data directory not found at:\n{self.__data_dir}'
            )

        if num_frames < 0:
            raise ValueError('Number of frames cannot be negative.')

        with open(self.__data_dir, 'r') as file:
            header = file.readline().strip().split(',')

        if any(
            col not in header
            for col in ['frame_number', 'pos_x', 'pos_y', 'pos_z']
        ):
            raise ValueError(
                'Data file must contain columns named frame_number, pos_x, pos_y, pos_z.'
            )

        dtype_map = [
            (
                name,
                (
                    'f8'
                    if name in ['pos_x', 'pos_y', 'pos_z']
                    else 'i8'
                    if name == 'frame_number'
                    else 'U32'
                ),
            )
            for name in header
        ]

        # read in data now that columns have been validated and typed
        frames = np.genfromtxt(
            self.__data_dir, delimiter=',', dtype=dtype_map, skip_header=1
        )

        # Rescale position data (e.g., convert meters to millimeters)
        if self.__rescale_by <= 0.0:
            raise ValueError('Rescale factor must be positive')

        # TODO: make this a param
        for col in ['pos_x', 'pos_y', 'pos_z']:
            frames[col][:] = frames[col][:] * self.__rescale_by

        if num_frames == 0:
            num_frames = self.__window_size

        # Calculate which frames to include
        last_frame = frames['frame_number'][-1]
        lookback = last_frame - num_frames

        # Filter for relevant frames
        frames = frames[frames['frame_number'] > lookback]

        return frames
