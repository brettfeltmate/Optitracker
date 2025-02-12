# Copyright © 2018 Naturalpoint
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# OptiTrack NatNet direct depacketization library for Python 3.x

import socket
import struct
import time
from threading import Thread
from typing import Any, Callable, Union

from optitracker.modules.parser.DataParser import DataParser as Parser


def trace(*args):
    # uncomment the one you want to use
    pass
    # print(''.join(map(str, args)))


def trace_dd(*args):
    # uncomment the one you want to use
    pass
    # print(''.join(map(str, args)))


def trace_mf(*args):
    # uncomment the one you want to use
    pass
    # print(''.join(map(str, args)))


def get_message_id(bytestream: bytes) -> int:
    message_id = int.from_bytes(bytestream[0:2], byteorder='little')
    return message_id


class NatNetClient:
    print_level = 0

    def __init__(
        self, instance_settings: dict[str, Union[str, int, bool]] = {}
    ) -> None:

        self.settings = {
            'server_ip': '127.0.0.1',
            # Change this value to the IP address of your local network interface
            'local_ip': '127.0.0.1',
            # This should match the multicast address listed in Motive's streaming settings.
            'multicast': '239.255.42.99',
            # NatNet Command channel
            'command_port': 1510,
            # NatNet Data channel
            'data_port': 1511,
            'use_multicast': True,
            # Set Application Name
            'apllication_name': 'Not Set',
            # NatNet stream version server is capable of. This will be updated during initialization only.
            'server_stream_version': [0, 0, 0, 0],
            # NatNet stream version. This will be updated to the actual version the server is using during runtime.
            'requested_natnet_version': [0, 0, 0, 0],
            # server stream version. This will be updated to the actual version the server is using during initialization.
            'server_version': [0, 0, 0, 0],
            # Lock values once run is called
            'is_locked': False,
            # Server has the ability to change bitstream version
            'can_change_bitstream_version': False,
        }

        self.settings.update(instance_settings)

        self.listeners = {
            'prefix': None,
            'marker': None,
            'rigid_body': None,
            'labeled_markers': None,
            'legacy_markers': None,
            'skeletons': None,
            'asset_rigid_bodies': None,
            'asset_markers': None,
            'channels': None,
            'force_plates': None,
            'devices': None,
            'suffix': None,
        }

        self.command_thread = None
        self.data_thread = None
        self.command_socket = None
        self.data_socket = None

        self.stop_threads = False

        # Constants corresponding to Client/server message ids
        self.message_ids = {
            'NAT_CONNECT': 0,
            'NAT_SERVERINFO': 1,
            'NAT_REQUEST': 2,
            'NAT_RESPONSE': 3,
            'NAT_REQUEST_MODELDEF': 4,
            'NAT_MODELDEF': 5,
            'NAT_REQUEST_FRAMEOFDATA': 6,
            'NAT_FRAMEOFDATA': 7,
            'NAT_MESSAGESTRING': 8,
            'NAT_DISCONNECT': 9,
            'NAT_KEEPALIVE': 10,
            'NAT_UNRECOGNIZED_REQUEST': 100,
            'NAT_UNDEFINED': 999999.9999,
        }

    def __unpack_data(self, stream: bytes) -> int:
        parse = Parser(stream=stream)
        frame_number = parse.frame_number()

        n_marker_sets, _ = parse.count()
        _, _ = parse.bytelen()

        # TODO: Pointer() might aide skipping
        for _ in range(0, n_marker_sets):
            set_label, _ = parse.label()

            marker_set = {'label': set_label, 'markers': []}

            n_markers_in_set, _ = parse.count()

            if self.listeners['marker'] is not None:
                for _ in range(n_markers_in_set):
                    marker, _ = parse.struct('unlabeled_marker')
                    marker['frame_number'] = frame_number  # type: ignore
                    marker_set['markers'].append(marker)

                self.listeners['marker'](marker_set)

            else:
                parse.seek(by=parse.size('marker', n_markers_in_set))

        n_rigid_bodies, _ = parse.count()
        _, _ = parse.bytelen()

        if self.listeners['rigid_body'] is not None:

            for _ in range(n_rigid_bodies):
                rigid_body = parse.struct('rigid_body')
                rigid_body['frame_number'] = frame_number  # type: ignore
                self.listeners['rigid_body'](rigid_body)
        else:
            parse.seek(parse.size('rigid_body', n_rigid_bodies))

        return parse.tell()

        # def __unpack_descriptions(self, bytestream: bytes):

        # Expected stream structure:
        # Int32ul = num of asset descriptions
        # Int32ul = asset type
        # Int32ul = asset description size
        # remaining = asset descriptions

        # Asset types:
        # 0 = MarkerSet
        # 1 = RigidBody
        # 2 = Skeleton
        # 3 = ForcePlate
        # 4 = Device
        # 5 = Camera
        # 6 = "Asset" whatever that is

        # pass

    def __unpack_bitstream_info(self, bytestream: bytes) -> list[str]:
        nn_version = []
        inString = bytestream.decode('utf-8')
        messageList = inString.split(',')
        if len(messageList) > 1:
            if messageList[0] == 'Bitstream':
                nn_version = messageList[1].split('.')
        return nn_version

    def __unpack_server_info(self, bytestream: bytes, offset: int) -> int:
        # Server name
        self.settings['application_name'], _, _ = bytes(
            bytestream[offset : offset + 256]
        ).partition(b'\0')
        self.settings['application_name'] = str(
            self.settings['application_name'], 'utf-8'
        )

        # Server Version info
        self.settings['server_version'] = struct.unpack(
            'BBBB', bytestream[offset + 256 : offset + 260]
        )

        # NatNet Version info
        self.settings['server_stream_version'] = struct.unpack(
            'BBBB', bytestream[offset + 260 : offset + 264]
        )

        if self.settings['requested_natnet_version'][:2] == [0, 0]:
            print(
                f"Resetting requested version to {self.settings['server_stream_version']} from {self.settings['nat_net_requested_version']}"
            )
            self.settings['requested_natnet_version'] = self.settings[
                'server_stream_version'
            ]
            # Determine if the bitstream version can be changed
            self.settings['can_change_bitstream_version'] = (
                self.settings['server_stream_version'][0] >= 4
                and not self.settings['use_multicast']
            )

        trace_mf(
            f"Sending Application Name: {self.settings['application_name']}"
        )
        trace_mf(f"NatNetVersion: {self.settings['server_stream_version']}")
        trace_mf(f"ServerVersion: {self.settings['server_version']}")
        return offset + 264

    def __create_command_socket(self) -> Union[socket.socket, None]:
        try:
            if self.settings['use_multicast']:
                result = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
                result.bind(('', 0))
                result.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            else:  # using unicast
                result = socket.socket(
                    socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
                )
                result.bind((self.settings['local_ip'], 0))

            # common settings for both cases
            result.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            result.settimeout(
                2.0
            )  # set timeout to allow for keep alive messages
            return result

        except (socket.herror, socket.gaierror):
            print(
                'error: command socket herror or gaierror occurred.\n'
                + 'check motive/server mode requested mode agreement.\n'
                + f'you requested {"multicast" if self.settings["use_multicast"] else "unicast"}'
            )
        except socket.timeout:
            print(
                'error: command socket timeout occurred. server not responding'
            )

        return None

    def __create_data_socket(self, port: int) -> Union[socket.socket, None]:
        try:
            result = socket.socket(
                socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
            )
            result.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            if self.settings['use_multicast']:
                # multicast case
                result.setsockopt(
                    socket.IPPROTO_IP,
                    socket.IP_ADD_MEMBERSHIP,
                    socket.inet_aton(self.settings['multicast'])
                    + socket.inet_aton(self.settings['local_ip']),
                )
                result.bind((self.settings['local_ip'], port))
            else:
                # unicast case
                result.bind(('', 0))
                if self.settings['multicast'] != '255.255.255.255':
                    result.setsockopt(
                        socket.IPPROTO_IP,
                        socket.IP_ADD_MEMBERSHIP,
                        socket.inet_aton(self.settings['multicast'])
                        + socket.inet_aton(self.settings['local_ip']),
                    )

            return result

        except (socket.herror, socket.gaierror):
            print(
                'error: data socket herror or gaierror occurred.\n'
                + 'check motive/server mode requested mode agreement.\n'
                + f'you requested {"multicast" if self.settings["use_multicast"] else "unicast"}'
            )
        except socket.timeout:
            print(
                'error: command socket timeout occurred. server not responding'
            )

        return None

    def __command_thread_callback(
        self, in_socket: socket.socket, stop: Callable, gprint_level: int = 1
    ) -> int:
        message_id_dict = {}
        if not self.settings['use_multicast']:
            in_socket.settimeout(2.0)

        # 64k buffer size
        recv_buffer_size = 64 * 1024
        while not stop():
            # Block for input
            try:
                bytestream, _ = in_socket.recvfrom(recv_buffer_size)
            except (
                socket.error,
                socket.herror,
                socket.gaierror,
                socket.timeout,
            ) as e:
                if (
                    stop()
                    or isinstance(e, socket.timeout)
                    and self.settings['use_multicast']
                ):
                    print(f'ERROR: command socket access error occurred:\n{e}')
                if isinstance(e, socket.error):
                    print('shutting down')
                return 1

            if bytestream:
                # peek ahead at message_id
                message_id = get_message_id(bytestream)
                tmp_str = f'mi_{message_id:.1f}'
                message_id_dict[tmp_str] = message_id_dict.get(tmp_str, 0) + 1

                # print_level = gprint_level()
                if (
                    message_id == self.message_ids['NAT_FRAMEOFDATA']
                    and gprint_level > 0
                ):
                    _ = (
                        1
                        if message_id_dict[tmp_str] % gprint_level == 0
                        else 0
                    )

                message_id = self.__process_message(bytestream)
                bytestream = bytearray()

            if not self.settings['use_multicast'] and not stop():
                self.send_keep_alive(
                    in_socket,
                    self.settings['server_ip'],
                    self.settings['command_port'],
                )

        return 0

    def __data_thread_callback(
        self, in_socket: socket.socket, stop: Callable, gprint_level: Callable
    ) -> int:
        message_id_dict = {}
        # 64k buffer size
        recv_buffer_size = 64 * 1024

        while not stop():
            # Block for input
            try:
                bytestream, _ = in_socket.recvfrom(recv_buffer_size)
            except (
                socket.error,
                socket.herror,
                socket.gaierror,
                socket.timeout,
            ) as e:
                if not stop() or isinstance(e, socket.timeout):
                    print(f'ERROR: data socket access error occurred:\n{e}')
                return 1

            if bytestream:
                # peek ahead at message_id
                message_id = get_message_id(bytestream)
                tmp_str = f'mi_{message_id:.1f}'
                message_id_dict[tmp_str] = message_id_dict.get(tmp_str, 0) + 1

                print_level = gprint_level()
                if (
                    message_id == self.message_ids['NAT_FRAMEOFDATA']
                    and print_level > 0
                ):
                    print_level = (
                        1 if message_id_dict[tmp_str] % print_level == 0 else 0
                    )

                message_id = self.__process_message(bytestream)
                bytestream = bytearray()

        return 0

    def __process_message(self, bytestream: bytes) -> int:
        message_id = get_message_id(bytestream)
        packet_size = int.from_bytes(bytestream[2:4], byteorder='little')

        # skip the 4 bytes for message ID and packet_size
        offset = 4
        if message_id == self.message_ids['NAT_FRAMEOFDATA']:
            offset += self.__unpack_data(bytestream[offset:])

        elif message_id == self.message_ids['NAT_MODELDEF']:
            pass
            # offset += self.__unpack_descriptions(bytestream[offset:])

        elif message_id == self.message_ids['NAT_SERVERINFO']:
            trace(
                f'Message ID: {message_id:.1f} {self.message_ids["NAT_SERVERINFO"]}, packet size: {packet_size}'
            )
            offset += self.__unpack_server_info(bytestream, offset)

        elif message_id in [
            self.message_ids['NAT_RESPONSE'],
            self.message_ids['NAT_UNRECOGNIZED_REQUEST'],
            self.message_ids['NAT_MESSAGESTRING'],
        ]:
            offset = self.__process_response(
                bytestream[offset:], packet_size, message_id
            )

        else:
            trace(f'Message ID: {message_id:.1f} (UNKNOWN)')
            trace(f'ERROR: Unrecognized packet type of size: {packet_size}')

        trace('End Packet\n-----------------')
        return message_id

    def __process_response(
        self, bytestream: bytes, packet_size: int, message_id: int
    ) -> int:
        offset = 0
        if message_id == self.message_ids['NAT_RESPONSE']:
            if packet_size == 4:
                command_response = int.from_bytes(
                    bytestream[offset : offset + 4], byteorder='little'
                )
                trace(
                    f'Command response: {command_response} - {[bytestream[offset+i] for i in range(4)]}'
                )
                offset += 4
            else:
                message, _, _ = bytes(bytestream[offset:]).partition(b'\0')
                if message.decode('utf-8').startswith('Bitstream'):
                    nn_version = self.__unpack_bitstream_info(
                        bytestream[offset:]
                    )
                    # Update the server version
                    self.settings['server_stream_version'] = [
                        int(v) for v in nn_version
                    ] + [0] * (4 - len(nn_version))
                trace(f"Command response: {message.decode('utf-8')}")
                offset += len(message) + 1
        elif message_id == self.message_ids['NAT_UNRECOGNIZED_REQUEST']:
            trace(
                f'Message ID:{message_id:.1f} {self.message_ids["NAT_UNRECOGNIZED_REQUEST"]}'
            )
            trace(f'Packet Size: {packet_size}')
        elif message_id == self.message_ids['NAT_MESSAGESTRING']:
            trace(
                f'Message ID:{message_id:.1f} {self.message_ids["NAT_MESSAGESTRING"]}, Packet Size: {packet_size}'
            )
            # create a command socket to attach to the natnet stream
            message, _, _ = bytes(bytestream[offset:]).partition(b'\0')
            trace(
                f"\n\tReceived message from server: {message.decode('utf-8')}"
            )
            offset += len(message) + 1

        return offset

    def connected(self) -> bool:
        return not (
            self.command_socket is None
            or self.data_socket is None
            or self.settings['application_name'] == 'not set'
            or self.settings['server_version'] == [0, 0, 0, 0]
        )

    def send_request(
        self,
        in_socket: socket.socket,
        command: int,
        command_str: str | list[int],
        address: tuple[Any, ...],
    ):
        if command in [
            self.message_ids['NAT_REQUEST_MODEDEF'],
            self.message_ids['NAT_REQUEST_FRAMEOFDATA'],
            self.message_ids['NAT_KEEPALIVE'],
        ]:
            packet_size = 0
        else:
            packet_size = len(command_str) + 1

        data = command.to_bytes(2, byteorder='little') + packet_size.to_bytes(
            2, byteorder='little'
        )

        if command == self.message_ids['NAT_CONNECT']:
            command_str = [80, 105, 110, 103] + [0] * 260 + [4, 1, 0, 0]
            print(f'NAT_CONNECT to Motive with {command_str[-4:]}\n')
            data += bytearray(command_str)
        else:
            data += command_str.encode('utf-8')  # type: ignore

        data += b'\0'
        return in_socket.sendto(data, address)

    def send_command(self, command_str: str) -> int:
        # print("Send command %s"%command_str)
        nTries = 3
        ret_val = -1
        for _ in range(nTries):
            ret_val = self.send_request(
                self.command_socket,  # type: ignore
                self.message_ids['NAT_REQUEST'],
                command_str,
                (self.settings['server_ip'], self.settings['command_port']),
            )
            if ret_val != -1:
                break
        return ret_val

    def send_commands(
        self, tmpcommands: list[str], print_results: bool = True
    ) -> None:

        for sz_command in tmpcommands:
            return_code = self.send_command(sz_command)
            if print_results:
                print(
                    'command: %s - return_code: %d' % (sz_command, return_code)
                )

    def send_keep_alive(
        self,
        in_socket: socket.socket,
        server_ip_address: str,
        server_port: int,
    ):
        return self.send_request(
            in_socket,
            self.message_ids['NAT_KEEPALIVE'],
            '',
            (server_ip_address, server_port),
        )

    def refresh_configuration(self) -> None:
        # query for application configuration
        # print("Request current configuration")
        sz_command = 'Bitstream'
        _ = self.send_command(sz_command)
        time.sleep(0.5)

    def startup(self) -> bool:
        # Create the data socket
        self.data_socket = self.__create_data_socket(
            self.settings['data_port']
        )
        if self.data_socket is None:
            print('Could not open data channel')
            return False

        # Create the command socket
        self.command_socket = self.__create_command_socket()
        if self.command_socket is None:
            print('Could not open command channel')
            return False
        self.settings['is_locked'] = True

        self.stop_threads = False
        # Create a separate thread for receiving data packets
        self.data_thread = Thread(
            target=self.__data_thread_callback,
            args=(
                self.data_socket,
                lambda: self.stop_threads,
                lambda: self.print_level,
            ),
        )
        self.data_thread.start()

        # Create a separate thread for receiving command packets
        self.command_thread = Thread(
            target=self.__command_thread_callback,
            args=(
                self.command_socket,
                lambda: self.stop_threads,
                lambda: self.print_level,
            ),
        )
        self.command_thread.start()

        # Required for setup
        # Get NatNet and server versions
        self.send_request(
            self.command_socket,
            self.message_ids['NAT_CONNECT'],
            '',
            (self.settings['server_ip'], self.settings['command_port']),
        )

        self.send_request(
            self.command_socket,
            self.message_ids['NAT_REQUEST_FRAMEOFDATA'],
            '',
            (self.settings['server_ip'], self.settings['command_port']),
        )
        return True

    def shutdown(self) -> None:
        print('shutdown called')
        self.stop_threads = True
        # closing sockets causes blocking recvfrom to throw
        # an exception and break the loop
        self.command_socket.close()  # type: ignore
        self.data_socket.close()  # type: ignore
        # attempt to join the threads back.
        self.command_thread.join()   # type: ignore
        self.data_thread.join()   # type: ignore
