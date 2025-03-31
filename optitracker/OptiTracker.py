from threading import Thread
import time
import os
import numpy as np
from csv import DictWriter
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
        rescale_by: int = 1000,
        init_natnet: bool = True,
        primary_axis: str = 'z',
        use_mouse: bool = False,
        display_ppi: int = -1,
    ):
        """Initialize the OptiTracker object.

        Args:
            marker_count (int): Number of markers being tracked
            sample_rate (int, optional): Data sampling rate in Hz. Defaults to 120.
            window_size (int, optional): Number of frames for temporal calculations. Defaults to 5.
            data_dir (str, optional): Path to the tracking data file. Defaults to "".
            rescale_by (float, optional): Factor to rescale position values. Defaults to 1000.
            init_natnet (bool, optional): Whether to initialize NatNet client. Defaults to True.
            primary_axis (str): Primary axis of movement (e.g., 'x', 'y', 'z')

        Raises:
            ValueError: If marker_count is non-positive integer
            ValueError: If sample_rate is non-positive integer
            ValueError: If window_size is non-positive integer
            ValueError: If rescale_by is non-positive numeric
            ValueError: If primary_axis is None or empty
            ValueError: If primary_axis is not 'x', 'y', or 'z'
        """
        self.console = Console()

        self.__is_listening = False

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

        self.__use_mouse = use_mouse

        if self.__use_mouse:
            import pyautogui

            init_natnet = False
            self.__natnet = None

            if display_ppi <= 0:
                self.__display_ppi = display_ppi
            else:
                raise ValueError(
                    'Display PPI must be specified for mouse tracking.'
                )

            _, self.__screen_height = pyautogui.size()
            self.__data_dir = 'mouse_tracking.csv'
            self.__mouse_thread = None
            self.__stop_mouse_thread = False

            if os.path.exists(self.__data_dir):
                os.remove(self.__data_dir)

        if init_natnet:
            self.__natnet = NatNetClient()
            self.__natnet.listeners['marker'] = self.__write  # type: ignore

        if primary_axis == '' or primary_axis is None:
            raise ValueError('Primary axis must be specified.')

        if primary_axis not in ['x', 'y', 'z', 'all']:
            raise ValueError('Primary axis must be one of: x, y, z, all')

        self.__primary_axis = primary_axis

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
    def rescale_by(self) -> int:
        """Get the rescaling factor used to convert position data.

        This factor is multiplied with all position values after reading from file.
        For example, use 1000 to convert meters to millimeters.
        """
        return self.__rescale_by

    @property
    def window_size(self) -> int:
        """Get the window size."""
        return self.__window_size

    def __get_mouse_position(self) -> np.ndarray:
        self.__mouse_frame += 1

        import pyautogui

        frame = np.ndarray(
            1,
            dtype=[
                ('frame_number', 'i8'),
                ('pos_x', 'f8'),
                ('pos_y', 'f8'),
                ('pos_z', 'f8'),
            ],
        )

        raw_pos = pyautogui.position()

        frame['frame_number'][:] = self.__mouse_frame
        frame['pos_x'][:] = float(raw_pos[0]) / (self.__display_ppi / 25.4)
        frame['pos_y'][:] = float(0)
        frame['pos_z'][:] = self.__screen_height - float(raw_pos[1]) / (
            self.__display_ppi / 25.4
        )

        return frame

    def is_listening(self) -> bool:
        """Return whether the NatNet client is currently listening."""
        return self.__is_listening

    def start_listening(self) -> bool:
        """
        Start listening for NatNet data.

        Raises:
            ValueError: If data directory is unset
        """
        if self.__data_dir == '':
            raise ValueError('No data directory was set.')

        if self.__use_mouse:
            self.__mouse_frame = 0

            self.__stop_mouse_thread = False
            self.__mouse_thread = Thread(target=self.__track_mouse)
            self.__mouse_thread.start()
            self.__is_listening = True

        else:
            if self.__natnet is None:
                raise RuntimeError('NatNet client not initialized.')
            else:
                self.__is_listening = self.__natnet.startup()

        return self.__is_listening

    def stop_listening(self) -> None:
        """Stop listening for NatNet data."""

        if not self.__use_mouse:
            if self.__natnet is None:
                raise RuntimeError('NatNet client not initialized.')

            if self.__is_listening:
                self.__natnet.shutdown()
                self.__is_listening = False
        else:

            self.__stop_mouse_thread = True

            if self.__mouse_thread is not None:
                self.__mouse_thread.join()
                self.__mouse_thread = None

            if os.path.exists(self.__data_dir):
                os.remove(self.__data_dir)

            self.__is_listening = False

    def query_frames(self, num_frames: int = 0) -> np.ndarray:
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
        return self.__read(
            num_frames=num_frames,
        )

    def velocity(
        self,
        num_frames: int = 0,
        axis: str = 'z',
    ) -> np.float64:
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

        frames = self.__read(num_frames)

        velocities = self.__calc_vector_velocity(frames, axis)

        return np.mean(velocities['velocity'], dtype=np.float64)  # type: ignore

    def position(
        self,
    ) -> np.ndarray:
        """Calculates and returns mean position(s) for the last n frames.

        Returns:
            np.ndarray: Structured array containing the mean position with fields:
                - frame_number (int): Frame identifier
                - pos_x (float): X coordinate
                - pos_y (float): Y coordinate
                - pos_z (float): Z coordinate
        """
        frame = self.__read(num_frames=1)

        return self.__calc_position(frames=frame)

    def distance(self, num_frames: int = 0, axis: str = 'z') -> np.float64:
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

        frames = self.__read(num_frames)

        distances = self.__calc_vector_distance(frames, axis)

        return np.sum(distances['distance'], dtype=np.float64)   # type: ignore

    def __calc_vector_velocity(
        self, frames: np.ndarray = np.array([]), axis: str = 'z'
    ) -> np.ndarray:
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
            frames = self.__read()

        distances = self.__calc_vector_distance(frames, axis)
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
        self, frames: np.ndarray = np.array([]), axis: str = 'z'
    ) -> np.ndarray:
        """
        Calculate Euclidean distance between first and last frames.

        Args:
            frames (np.ndarray, optional): Array of frame data; queries last window_size frames if empty.
            axis (str, optional): Along which axis to calculate velocity; defaults to None (uses self.__primary_axis)

        Returns:
            float: Euclidean distance in mm
        """
        positions = self.__calc_position(frames)
        distances = np.ndarray(
            len(positions) - 1,
            dtype=[
                ('frame_number', 'i8'),
                ('distance', 'f8'),
            ],
        )

        # if axis is not None and axis not in ['x', 'y', 'z', 'all']:
        #     raise ValueError('Axis must be one of: x, y, z, all')

        if axis is None:
            axis = self.__primary_axis

        distances['frame_number'][:] = positions['frame_number'][1:]
        for i in range(len(positions) - 1):
            distances['frame_number'][i] = positions['frame_number'][i + 1]

            if axis == 'all':
                variance = np.sum(
                    [
                        np.diff(positions['pos_x'][i : i + 2]) ** 2,
                        np.diff(positions['pos_y'][i : i + 2]) ** 2,
                        np.diff(positions['pos_z'][i : i + 2]) ** 2,
                    ]
                )
                deviation = np.sqrt(variance)
                # TODO: retain sign of deviation

            else:
                deviation = np.diff(positions[f'pos_{axis}'][i : i + 2])

            distances['distance'][i] = deviation

        if axis:
            pass

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
        if self.__use_mouse:
            return frames

        if len(frames) == 0:
            frames = self.__read()

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

        return positions

    # def __validate_data(self, data: np.ndarray) -> None:
    #     """Validate the format of the data array.
    #
    #     Args:
    #         data (np.ndarray): Array of frame data to validate
    #
    #     Raises:
    #         ValueError: If data is empty
    #         ValueError: If data is not a structured array
    #         ValueError: If data does not contain expected fields
    #         ValueError: If datafields are not of expected types
    #     """
    #     if len(data) == 0:
    #         raise ValueError('No data was provided.')
    #
    #     if not data.dtype.names:
    #         raise ValueError('Data must be a structured array.')
    #
    #     if any(
    #         col not in data.dtype.names
    #         for col in ['frame_number', 'pos_x', 'pos_y', 'pos_z']
    #     ):
    #         raise ValueError(
    #             'Data must contain fields: frame_number, pos_x, pos_y, pos_z.'
    #         )
    #
    #     if any(
    #         data[col].dtype not in ['f8', 'i8']
    #         for col in ['frame_number', 'pos_x', 'pos_y', 'pos_z']
    #     ):
    #         raise ValueError(
    #             'Data fields must be of types: float64 (for position) or int64 (for frame number).'
    #         )

    def __read(self, num_frames: int = 0) -> np.ndarray:
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
            raise ValueError('Must specify data directory.')

        if not os.path.exists(self.__data_dir):
            raise FileNotFoundError(
                f'No data directory present at:\n{self.__data_dir}'
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
                'Data must contain columns frame_number, pos_x, pos_y, pos_z.'
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

        # self.__validate_data(frames)

        if not self.__use_mouse:
            # Rescale position data (e.g., convert meters to millimeters)
            if self.__rescale_by <= 0.0:
                raise ValueError('Rescale factor must be positive')

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

    def __write(self, frames: np.ndarray) -> None:
        """Write marker set data to CSV file.

        Args:
            marker_set (dict): Dictionary containing marker data to be written.
                Expected format: {'markers': [{'key1': val1, ...}, ...]}
        """
        if self.__use_mouse:
            frames = self.__get_mouse_position()
            fname = self.__data_dir
            header = list(frames.dtype.names)

            if not os.path.exists(fname):
                with open(self.__data_dir, 'w', newline='') as file:
                    writer = DictWriter(
                        file,
                        fieldnames=['frame_number', 'pos_x', 'pos_y', 'pos_z'],
                    )
                    writer.writeheader()

            with open(fname, 'a', newline='') as file:
                np.savetxt(file, frames, delimiter=',', fmt='%s')

        else:
            # if type(frames) is dict:
            if frames.get('label') == 'hand':
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
            # else:
            #     raise ValueError(
            #         'Frames of unexpected type. Should be dict or np.ndarray'
            #     )

    def __track_mouse(self) -> None:
        """Continuously track and write mouse position data."""

        while not self.__stop_mouse_thread:
            # Get and write mouse position
            self.__write(None)
            # Control sampling rate
            time.sleep(1.0 / self.__sample_rate)
