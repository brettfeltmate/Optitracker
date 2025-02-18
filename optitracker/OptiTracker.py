import os
import numpy as np
from csv import DictWriter
from scipy.signal import butter, sosfiltfilt
from rich.console import Console

from ..NatNetClient.NatNetClient import NatNetClient

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
        self.__console_logging = console_logging
        if self.__console_logging:
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

        self.__smooth_data = smooth_data

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
        """Get the rescaling factor used to convert position data.

        This factor is multiplied with all position values after reading from file.
        For example, use 1000 to convert meters to millimeters.
        """
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

    def query_frames(
        self, num_frames: int = 0, console_logging: bool | None = None
    ) -> np.ndarray:
        """Query the most recent frames from the tracking data.

        Args:
            num_frames (int, optional): Number of frames to query. If 0, uses the instance's window_size. Defaults to 0.

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
        return self.__query_frames(
            num_frames=num_frames,
            console_logging=self.__console_logging or console_logging,
        )

    def velocity(
        self,
        num_frames: int = 0,
        smooth: bool | None = None,
        console_logging: bool | None = None,
    ) -> float:
        """Calculate the current velocity from position data.

        Computes velocity by measuring displacement over time using the specified
        number of frames or the default window size.

        Args:
            num_frames (int, optional): Number of frames to use for calculation.
                If 0, uses the instance's window_size. Defaults to 0.
            smooth (bool, optional): Whether to apply Butterworth smoothing. Defaults to None.

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

        return self.__velocity(
            frames=frames,
            smooth=self.__smooth_data or smooth,
            console_logging=self.__console_logging or console_logging,
        )

    def position(
        self,
        num_frames: int = 1,
        smooth: bool | None = None,
        console_logging: bool | None = None,
    ) -> np.ndarray:
        """Calculates and returns mean position(s) for the last n frames.

        Args:
            num_frames (int, optional): Number of frames to calculate mean position over. Defaults to 1.

        Returns:
            np.ndarray: Structured array containing the mean position with fields:
                - frame_number (int): Frame identifier
                - pos_x (float): X coordinate
                - pos_y (float): Y coordinate
                - pos_z (float): Z coordinate
        """
        frames = self.__query_frames(num_frames=num_frames)

        return self.__column_means(
            frames=frames,
            smooth=self.__smooth_data or smooth,
            console_logging=self.__console_logging or console_logging,
        )

    def distance(
        self,
        num_frames: int = 0,
        smooth: bool | None = None,
        console_logging: bool | None = None,
    ) -> float:
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

        frames = self.__query_frames(num_frames)

        return self.__euclidean_distance(
            frames=frames,
            smooth=self.__smooth_data or smooth,
            console_logging=self.__console_logging or console_logging,
        )

    def __velocity(
        self,
        frames: np.ndarray = np.array([]),
        smooth: bool | None = None,
        console_logging: bool | None = None,
    ) -> float:
        """
        Calculate velocity using position data over the specified window.

        Args:
            frames (np.ndarray, optional): Array of frame data; queries last window_size frames if empty.

        Returns:
            float: Calculated velocity in units/second (based on rescale_by factor)
        """
        if self.__window_size < 2:
            raise ValueError('Window size must cover at least two frames.')

        if len(frames) == 0:
            frames = self.__query_frames()

        # Get mean positions for each frame (averaging across markers)
        positions = self.__column_means(frames=frames, smooth=self.__smooth_data or smooth)
        
        # Calculate instantaneous velocities between consecutive frames
        velocities = []
        for i in range(len(positions) - 1):
            # Calculate displacement between consecutive frames
            dx = positions['pos_x'][i+1] - positions['pos_x'][i]
            dy = positions['pos_y'][i+1] - positions['pos_y'][i]
            dz = positions['pos_z'][i+1] - positions['pos_z'][i]
            
            # Calculate Euclidean distance for this step
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # Time between frames
            dt = 1.0 / self.__sample_rate
            
            # Calculate instantaneous velocity
            velocities.append(distance / dt)

        if self.__console_logging or console_logging:
            print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n__velocity()\n')
            self.console.log(log_locals=True)
            print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

        # Return mean velocity over the window
        return float(np.mean(velocities))

    def __euclidean_distance(
        self,
        frames: np.ndarray = np.array([]),
        smooth: bool | None = None,
        console_logging: bool | None = None,
    ) -> float:
        """
        Calculate Euclidean distance between first and last frames.

        Args:
            frames (np.ndarray, optional): Array of frame data; queries last window_size frames if empty.

        Returns:
            float: Euclidean distance
        """

        if frames.size == 0:
            frames = self.__query_frames()

        positions = self.__column_means(
            frames=frames, smooth=self.__smooth_data or smooth
        )

        if self.__console_logging or console_logging:
            print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n__smooth()\n')
            self.console.log(log_locals=True)
            print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

        return float(
            np.sqrt(
                (positions['pos_x'][-1] - positions['pos_x'][0]) ** 2
                + (positions['pos_y'][-1] - positions['pos_y'][0]) ** 2
                + (positions['pos_z'][-1] - positions['pos_z'][0]) ** 2
            )
        )

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
        if len(frames) == 0:
            frames = self.__query_frames()

        # Create output array with the correct dtype
        smoothed = np.zeros(
            len(frames),
            dtype=[
                ('frame_number', 'i8'),
                ('pos_x', 'f8'),
                ('pos_y', 'f8'),
                ('pos_z', 'f8'),
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
        smoothed['pos_x'][:] = sosfiltfilt(sos=butt, x=frames['pos_x'][:])
        smoothed['pos_y'][:] = sosfiltfilt(sos=butt, x=frames['pos_y'][:])
        smoothed['pos_z'][:] = sosfiltfilt(sos=butt, x=frames['pos_z'][:])

        if self.__console_logging or console_logging:
            print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n__smooth()\n')
            self.console.log(log_locals=True)
            print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

        return smoothed

    def __column_means(
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
            frames = self.__query_frames()

        # Create output array with the correct dtype
        means = np.zeros(
            len(frames) // self.__marker_count,
            dtype=[
                ('frame_number', 'i8'),
                ('pos_x', 'f8'),
                ('pos_y', 'f8'),
                ('pos_z', 'f8'),
            ],
        )

        # Group by marker (every nth row where n is marker_count)
        idx = 0
        start = min(frames['frame_number'])
        stop = max(frames['frame_number']) + 1

        for frame_number in range(start, stop):

            frame = frames[
                frames['frame_number'] == frame_number,
            ]

            means[idx]['frame_number'] = frame_number
            means[idx]['pos_x'] = np.mean(frame['pos_x'])
            means[idx]['pos_y'] = np.mean(frame['pos_y'])
            means[idx]['pos_z'] = np.mean(frame['pos_z'])

            idx += 1

            # try:
            #
            #     means[idx]["pos_x"] = np.mean(frame["pos_x"])
            #     means[idx]["pos_y"] = np.mean(frame["pos_y"])
            #     means[idx]["pos_z"] = np.mean(frame["pos_z"])
            #
            #     idx += 1
            #
            # except RuntimeWarning:
            #     means[idx]["pos_x"] = 0.0
            #     means[idx]["pos_y"] = 0.0
            #     means[idx]["pos_z"] = 0.0

        if smooth or self.__smooth_data:
            means = self.__smooth(frames=means)

        if console_logging or self.__console_logging:
            print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n__column_means()\n')
            self.console.log(log_locals=True)
            print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

        return means

    def __query_frames(
        self, num_frames: int = 0, console_logging: bool | None = None
    ) -> np.ndarray:
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
            # coerce expected columns to float, int, string (default)
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
        data = np.genfromtxt(
            self.__data_dir, delimiter=',', dtype=dtype_map, skip_header=1
        )

        # Rescale position data (e.g., convert meters to millimeters)
        if self.__rescale_by <= 0.0:
            raise ValueError('Rescale factor must be positive')

        # TODO: make this a param
        for col in ['pos_x', 'pos_y', 'pos_z']:
            data[col][:] = data[col][:] * self.__rescale_by

        if num_frames == 0:
            num_frames = self.__window_size

        # Calculate which frames to include
        last_frame = data['frame_number'][-1]
        lookback = last_frame - num_frames

        # Filter for relevant frames
        data = data[data['frame_number'] > lookback]

        if self.__console_logging or console_logging:
            print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n__query_frames()\n')
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
