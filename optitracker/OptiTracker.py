import os
import numpy as np
from csv import DictWriter
from scipy.signal import butter, sosfiltfilt
from rich.console import Console

from optitracker.modules.natnetclient.NatNetClient import NatNetClient

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
        smooth_data: bool = False,
        init_natnet: bool = True,
        console_logging: bool = False,
    ):
        """Initialize the OptiTracker object.

        Args:
            marker_count (int): Number of markers being tracked
            sample_rate (int, optional): Data sampling rate in Hz. Defaults to 120.
            window_size (int, optional): Number of frames for temporal calculations. Defaults to 5.
            data_dir (str, optional): Path to the tracking data file. Defaults to "".
            rescale_by (float, optional): Factor to rescale position values. Defaults to 1000.
            smooth_data (bool, optional): Whether to apply Butterworth smoothing. Defaults to False.
            init_natnet (bool, optional): Whether to initialize NatNet client. Defaults to True.
            console_logging (bool, optional): Whether to enable console_logging mode. Defaults to False.

        Raises:
            ValueError: If marker_count is non-positive integer
            ValueError: If sample_rate is non-positive integer
            ValueError: If window_size is non-positive integer
            ValueError: If rescale_by is non-positive numeric
        """
        if sample_rate <= 0:
            raise ValueError('Sample rate must be positive.')
        if marker_count <= 0:
            raise ValueError('Marker count must be positive.')
        if window_size < 1:
            raise ValueError('Window size must be postively non-zero.')
        if rescale_by <= 0.0:
            raise ValueError('Rescale factor must be positive')

        self.__console_logging = console_logging
        self.__marker_count = marker_count
        self.__sample_rate = sample_rate
        self.__window_size = window_size
        self.__rescale_by = rescale_by
        self.__data_dir = data_dir
        self.__smooth_data = smooth_data

        if self.__console_logging:
            self.console = Console()

        if init_natnet:
            self.__natnet_listening = False
            self.__natnet = NatNetClient()
            self.__natnet.listeners['marker'] = self.__write_frames  # type: ignore
        else:
            self.__natnet = None

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
        """Get the rescaling factor applied to position data. """
        return self.__rescale_by

    @property
    def window_size(self) -> int:
        """Get the window size."""
        return self.__window_size

    @property
    def smooth_data(self) -> bool:
        """Get the smoothing status."""
        return self.__smooth_data

    @smooth_data.setter
    def smooth_data(self, smooth_data: bool) -> None:
        """Set the smoothing status."""
        self.__smooth_data = smooth_data

    def start_listening(self) -> bool:
        """
        Start listening for NatNet data.

        Raises:
            ValueError: If data directory is unset
        """
        if self.__natnet is None:
            raise RuntimeError('NatNet client not initialized.')

        if self.__data_dir == '':
            raise ValueError('No data directory was set.')

        if not self.__natnet_listening:
            self.__natnet_listening = self.__natnet.startup()

        return self.__natnet_listening

    def stop_listening(self) -> None:
        """Stop listening for NatNet data."""

        if self.__natnet is None:

            raise RuntimeError('NatNet client not initialized.')

        if self.__natnet_listening:
            self.__natnet.shutdown()
            self.__natnet_listening = False

    def frames(
        self,
        num_frames: int | None = None,
        console_logging: bool | None = None,
    ) -> np.ndarray:

        """Query the most recent frames from the tracking data.


        Args:

            num_frames (int, optional): Number of frames to query. If None (default), queries window_size frames.

        Returns:
            np.ndarray: Structured array of frame data with fields:
                - frame_number (int): Frame identifier
                - pos_x (float): X coordinate
                - pos_y (float): Y coordinate
                - pos_z (float): Z coordinate

        Raises:
            ValueError: If data directory is empty
            FileNotFoundError: If data file does not exist
            ValueError: If num_frames is negative
        """
        return self.__read_frames(
            num_frames=num_frames,
            console_logging=self.__console_logging or console_logging,
        )

    def position(
        self,
        frames: np.ndarray = np.array([]),
        smoothed: bool | None = None,
        num_frames: int | None = None,
        instantaneous: bool = False,
        axes: str | list[str] = 'all',
        console_logging: bool | None = None,
    ):
        """Compute the mean position for each of the last num_frames.

        Args:
            frames (np.ndarray, optional): Array of frame data. Defaults to querying self.__window_size if unprovided.
            smoothed (bool, optional): Whether to apply Butterworth smoothing. Defaults to self.__smooth_data.
            num_frames (int, optional): Number of frames to query. Defaults to self.__window_size.
            instantaneous (bool, optional): Whether to return the current position. Defaults to False (returns positions across window_size).
            axes (str | list[str], optional): Axes for which to return positions ('all', 'x', 'y', 'z'). Defaults to 'all'.
            console_logging (bool, optional): Whether to enable console_logging mode. Defaults to self.__console_logging.

        Returns:
            np.ndarray: Structured array of mean positions with fields:
                - frame_number (int): Frame identifier
                - pos_x (float): Mean X coordinate if 'all' or 'x' specified
                - pos_y (float): Mean Y coordinate if 'all' or 'y' specified
                - pos_z (float): Mean Z coordinate if 'all' or 'z' specified
        """

        if smoothed and num_frames == 1:
            raise ValueError('Cannot smooth a single frame.')

        if not isinstance(axes, list):
            axes = [axes]

        if not all(axis in ['all', 'x', 'y', 'z'] for axis in axes):
            raise ValueError(
                'Invalid axes specified. Must be "all", "x", "y", "z", or list thereof.'
            )

        if frames.size == 0:
            frames = self.__read_frames(num_frames=num_frames)

        positions = self.__compute(
            metric='position',
            smooth=smoothed,
            frames=frames,
            console_logging=self.__console_logging or console_logging,
        )

        if axes == 'all':
            return positions[-1, :] if instantaneous else positions
        else:
            return positions[-1, :][axes] if instantaneous else positions[axes]

    def distance(
        self,
        frames: np.ndarray = np.array([]),
        smoothed: bool | None = None,
        num_frames: int | None = None,
        instantaneous: bool = False,
        axes: str | list[str] = 'all',
        console_logging: bool | None = None,
    ) -> np.ndarray:
        """Compute the Euclidean distance between the first and last frames.

        Args:
            frames (np.ndarray, optional): Array of frame data. Defaults to querying self.__window_size if unprovided.
            smoothed (bool, optional): Whether to apply Butterworth smoothing. Defaults to self.__smooth_data.
            num_frames (int, optional): Number of frames to query. Defaults to self.__window_size.
            instantaneous (bool, optional): Whether to return the current distance. Defaults to False (returns distances across window_size).
            axes (str | list[str], optional): Axes for which to return distances ('all', 'x', 'y', 'z'). Defaults to 'all'.
            console_logging (bool, optional): Whether to enable console_logging mode. Defaults to self.__console_logging.

        Returns:
            np.ndarray: Structured array of euclidean distances with fields:
                - frame_number (int): Frame identifier
                - dx (float): X distance
                - dy (float): Y distance
                - dz (float): Z distance
        """
        if not isinstance(axes, list):
            axes = [axes]

        if not all(axis in ['all', 'x', 'y', 'z'] for axis in axes):
            raise ValueError(
                'Invalid axes specified. Must be "all", "x", "y", "z", or list thereof.'
            )

        if frames.size == 0:
            frames = self.__read_frames(num_frames=num_frames)

        distances = self.__compute(
            metric='distance',
            smooth=smoothed,
            frames=frames,
            console_logging=self.__console_logging or console_logging,
        )

        if axes == 'all':
            return distances[-1, :] if instantaneous else distances
        else:
            return distances[-1, :][axes] if instantaneous else distances[axes]

    def velocity(
        self,
        frames: np.ndarray = np.array([]),
        smoothed: bool | None = None,
        num_frames: int | None = None,
        instantaneous: bool = False,
        axes: str | list[str] = 'all',
        console_logging: bool | None = None,
    ) -> np.ndarray:
        """Compute the velocity of the asset across the last num_frames.

        Args:
            frames (np.ndarray, optional): Array of frame data. Defaults to querying self.__window_size if unprovided.
            smoothed (bool, optional): Whether to apply Butterworth smoothing. Defaults to self.__smooth_data.
            num_frames (int, optional): Number of frames to query. Defaults to self.__window_size.
            instantaneous (bool, optional): Whether to return the current velocity. Defaults to False (returns velocities across window_size).
            axes (str | list[str], optional): Axes for which to return distances ('all', 'x', 'y', 'z'). Defaults to 'all'.
            console_logging (bool, optional): Whether to enable console_logging mode. Defaults to self.__console_logging.

        Returns:
            np.ndarray: Structured array of velocities with fields:
                - frame_number (int): Frame identifier
                - vx (float): X velocity (if 'all' or 'x' specified)
                - vy (float): Y velocity (if 'all' or 'y' specified)
                - vz (float): Z velocity (if 'all' or 'z' specified)
        """

        if not isinstance(axes, list):
            axes = [axes]
        if not all(axis in ['all', 'x', 'y', 'z'] for axis in axes):
            raise ValueError(
                'Invalid axes specified. Must be "all", "x", "y", "z", or list thereof.'
            )

        if frames.size == 0:
            frames = self.__read_frames(num_frames=num_frames)

        velocities = self.__compute(
            metric='velocity',
            smooth=smoothed,
            frames=frames,
            console_logging=self.__console_logging or console_logging,
        )

        if axes == 'all':
            if not instantaneous:
                return velocities
            else:
                # Get final velocity vector
                v = velocities[-1]
                # Calculate magnitude
                magnitude = np.sqrt(v['vx'] ** 2 + v['vy'] ** 2 + v['vz'] ** 2)
                # Get dominant axis velocity
                dominant_axis = max(
                    ['vx', 'vy', 'vz'], key=lambda ax: abs(v[ax])
                )
                # Return signed magnitude
                return magnitude * np.sign(v[dominant_axis])
        else:
            return (
                velocities[-1, :][axes] if instantaneous else velocities[axes]
            )

    def __compute(
        self,
        metric: str,
        frames: np.ndarray = np.array([]),
        smooth: bool | None = None,
        console_logging: bool | None = None,
    ) -> np.ndarray:
        """Compute the specified metric for the most recent frames.

        Args:
            metric (str): Metric to compute ('position', 'distance', 'velocity')
            frames (np.ndarray, optional): Array of frame data. If empty, queries last window_size frames. Defaults to empty array.
            smooth (bool, optional): Whether to apply Butterworth smoothing. Defaults to False.
        """

        if metric not in ['position', 'distance', 'velocity']:
            raise ValueError('Invalid metric specified.')

        if frames.size == 0:
            raise ValueError('No frame data provided.')

        positions = self.__asset_position(
            frames=frames,
            smooth=smooth,
            console_logging=self.__console_logging or console_logging,
        )

        if metric == 'position':
            return positions

        distances = self.__asset_euclidean_distance(
            positions=positions,
            console_logging=self.__console_logging or console_logging,
        )

        if metric == 'distance':
            return distances

        velocities = self.__asset_velocity(
            distances=distances,
            console_logging=self.__console_logging or console_logging,
        )

        return velocities

    def __asset_velocity(
        self,
        distances: np.ndarray = np.array([]),
        console_logging: bool | None = None,
    ) -> np.ndarray:

        """Calculate velocity using speed data over the specified window.

        Args:
            distances (np.ndarray, optional): Array of euclidean distances.
            axis (str, optional): Axis along which to calculate velocity ('all', 'x', 'y', 'z'). Defaults to 'all'.

        Returns:
            np.ndarray: Array of velocities with fields:
                - frame_number (int): Frame identifier
                - velocity (float): Velocity along the specified axes
        """
        if distances.size == 0:
            raise ValueError('No distance data provided.')

        # Create output array with the correct dtype
        velocities = np.zeros(
            len(distances),
            dtype=[
                ('frame_number', 'i8'),
                ('vx', 'f8'),
                ('vy', 'f8'),
                ('vz', 'f8'),
            ],
        )

        velocities['vx'][:] = np.diff(distances['dx']) / (
            1.0 / self.__sample_rate
        )
        velocities['vy'][:] = np.diff(distances['dy']) / (
            1.0 / self.__sample_rate
        )
        velocities['vz'][:] = np.diff(distances['dz']) / (
            1.0 / self.__sample_rate
        )

        if self.__console_logging or console_logging:
            print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
            print('\n\n __velocity()\n\n')
            self.console.log(log_locals=True)
            print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

        return velocities[
            :,
        ]

    def __asset_euclidean_distance(
        self,
        positions: np.ndarray = np.array([]),
        console_logging: bool | None = None,
    ) -> np.ndarray:
        """
        Calculate Euclidean distance between first and last frames.

        Args:
            frames (np.ndarray, optional): Array of frame data; queries last window_size frames if empty.
            axis (str, optional): Axis along which to calculate distance ('all', 'x', 'y', 'z'). Defaults to 'all'.

        Returns:
            float: Euclidean distance
        """

        if positions.size == 0:
            raise ValueError('No frame data provided.')

        # Create np array of momentary euclidean distances
        distances = np.zeros(
            len(positions) - 1,
            dtype=[
                ('frame_number', 'i8'),
                ('dx', 'f8'),
                ('dy', 'f8'),
                ('dz', 'f8'),
            ],
        )

        # Calculate momentary distances
        distances['dx'][:] = np.diff(positions['px'])
        distances['dy'][:] = np.diff(positions['py'])
        distances['dz'][:] = np.diff(positions['pz'])

        if self.__console_logging or console_logging:
            print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
            print('\n\n __euclidean_distance()\n\n')
            self.console.log(log_locals=True)
            print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

        return distances

    # TODO: reduce dependencies by hand-rolling a butterworth filter

    def __smooth(
        self,
        frames: np.ndarray = np.array([]),
        order=2,
        cutoff=10,
        filtype='low',
        console_logging: bool | None = None,
    ) -> np.ndarray:
        """Apply a zero-phase Butterworth filter to position data.

        Uses scipy.signal.sosfiltfilt for zero-phase digital filtering, which
        processes the input data forwards and backwards to eliminate phase delay.

        Args:
            order (int, optional): Order of the Butterworth filter. Defaults to 2.
            cutoff (int, optional): Cutoff frequency in Hz. Defaults to 10.
            filtype (str, optional): Filter type ('low', 'high', 'band'). Defaults to "low".
            frames (np.ndarray, optional): Structured array of frame data.
                If empty, queries last window_size frames. Defaults to empty array.

        Returns:
            np.ndarray: Structured array of filtered positions with fields:
                - frame_number (int): Frame identifier
                - pos_x (float): Filtered X coordinate
                - pos_y (float): Filtered Y coordinate
                - pos_z (float): Filtered Z coordinate

        Note:
            The filter is applied separately to each position dimension
        """
        if frames.size == 0:
            raise ValueError('No frame data provided.')

        # Create output array with the correct dtype
        smoothed = np.zeros(
            len(frames),
            dtype=[
                ('frame_number', 'i8'),
                ('px', 'f8'),
                ('py', 'f8'),
                ('pz', 'f8'),
            ],
        )

        butt = butter(
            N=order,
            Wn=cutoff,
            btype=filtype,
            output='sos',
            fs=self.__sample_rate,
        )

        smoothed['frame_number'][:] = frames['frame_number'][:]
        smoothed['px'][:] = sosfiltfilt(sos=butt, x=frames['px'][:])
        smoothed['py'][:] = sosfiltfilt(sos=butt, x=frames['py'][:])
        smoothed['pz'][:] = sosfiltfilt(sos=butt, x=frames['pz'][:])

        if self.__console_logging or console_logging:
            print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
            print('\n\n __smooth()\n\n')
            self.console.log(log_locals=True)
            print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

        return smoothed

    def __asset_position(
        self,
        frames: np.ndarray = np.array([]),
        smooth: bool | None = None,
        console_logging: bool | None = None,
    ) -> np.ndarray:
        """Calculate mean positions across all markers for each frame.

        For each frame, computes the centroid position by averaging the positions
        of all markers tracked in that frame.

        Args:
            smooth (bool, optional): Whether to apply Butterworth filtering to means.
                Defaults to True.
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
            raise ValueError('No frame data provided.')

        # Create output array with the correct dtype
        positions = np.zeros(
            len(frames) // self.__marker_count,
            dtype=[
                ('frame_number', 'i8'),
                ('px', 'f8'),
                ('py', 'f8'),
                ('pz', 'f8'),
            ],
        )

        # Group by marker (every nth row where n is marker_count)
        idx = 0
        start = min(frames['frame_number'])
        stop = max(frames['frame_number']) + 1

        for frame_number in range(start, stop):

            this_frame = frames[
                frames['frame_number'] == frame_number,
            ]

            positions[idx]['frame_number'] = frame_number
            positions[idx]['px'] = np.mean(this_frame['pos_x'])
            positions[idx]['py'] = np.mean(this_frame['pos_y'])
            positions[idx]['pz'] = np.mean(this_frame['pos_z'])

            idx += 1

        if smooth or self.__smooth_data:
            positions = self.__smooth(frames=positions)

        if console_logging or self.__console_logging:
            print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
            print('\n\n __asset_position()\n\n')
            self.console.log(log_locals=True)
            print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

        return positions

    def __read_frames(
        self,
        num_frames: int | None = None,
        console_logging: bool | None = None,
    ) -> np.ndarray:
        """Load and process frame data from the tracking data file.

        Reads position data from CSV file, validates format, and applies rescaling.
        Returns the most recent frames up to the specified number.

        Args:
            num_frames (int, optional): Number of most recent frames to return.
                If None (default), defaults to the instance's window_size.

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


        # Set num_frames to window_size if not specified
        if num_frames is None:
            num_frames = self.__window_size

        # If provided, validate num_frames
        if num_frames < 0:
            raise ValueError('Number of frames cannot be negative.')

        # Validate data format
        with open(self.__data_dir, 'r') as file:
            header = file.readline().strip().split(',')

        if any(
            col not in header
            for col in ['frame_number', 'pos_x', 'pos_y', 'pos_z']
        ):
            raise ValueError(
                'Data file must contain columns named frame_number, pos_x, pos_y, pos_z.'
            )

        # Map column names to dtypes
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

        # read in data
        data = np.genfromtxt(
            self.__data_dir, delimiter=',', dtype=dtype_map, skip_header=1
        )

        # Rescale coordinate data
        for col in ['pos_x', 'pos_y', 'pos_z']:
            data[col][:] = data[col][:] * self.__rescale_by

        # Calculate which frames to include
        last_frame = data['frame_number'][-1]
        lookback = last_frame - num_frames

        # Filter for relevant frames
        data = data[data['frame_number'] > lookback]

        if self.__console_logging or console_logging:
            print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
            print('\n\n __read_frames()\n\n')
            self.console.log(log_locals=True)
            print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

        return data

    def __write_frames(self, set_name: str, frames: dict) -> None:
        """Write marker set data to CSV file.

        Args:
            marker_set (dict): Dictionary containing marker data to be written.
                Expected format: {'markers': [{'key1': val1, ...}, ...]}
        """

        if frames.get('label') == set_name:
            # Append data to trial-specific CSV file
            fname = self.__data_dir
            header = list(frames['markers'][0].keys())

            # if file doesn't exist, create it and write header
            if not os.path.exists(fname):
                with open(fname, 'w', newline='') as file:
                    writer = DictWriter(file, fieldnames=header)
                    writer.writeheader()

            # append marker data to file
            with open(fname, 'a', newline='') as file:
                writer = DictWriter(file, fieldnames=header)
                for marker in frames.get('markers', None):
                    if marker is not None:
                        writer.writerow(marker)
