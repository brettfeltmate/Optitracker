# type: ignore

from construct import this, Float32l, Int16sl, Struct, Computed, Int32ul, CString
from typing import Union, Container


def decodeMarkerID(obj, _):
    return obj.encoded_id & 0x0000FFFF


def decodeModelID(obj, _):
    return obj.encoded_id >> 16


def trackingValid(obj, _):
    return (obj.error & 0x01) != 0


unlabeledMarkerStruct = Struct(
    "pos_x" / Float32l,
    "pos_y" / Float32l,
    "pos_z" / Float32l,
)

labeledMarkerStruct = Struct(
    "id" / Int32ul,
    "marker_id" / Computed(this.id * decodeMarkerID),
    "model_id" / Computed(this.id * decodeModelID),
    "pos_x" / Float32l,
    "pos_y" / Float32l,
    "pos_z" / Float32l,
    "size" / Float32l,
    "param" / Int16sl,
    "residual" / Float32l,
)


rigidBodyStruct = Struct(
    "id" / Int32ul,
    "pos_x" / Float32l,
    "pos_y" / Float32l,
    "pos_z" / Float32l,
    "rot_w" / Float32l,
    "rot_x" / Float32l,
    "rot_y" / Float32l,
    "rot_z" / Float32l,
    "error" / Float32l,
    "tracking" / Int16sl,
    "is_valid" / Computed(lambda ctx: (ctx.tracking & 0x01) != 0),
)

class MotiveStreamParser(object):
    def __init__(self, stream: bytes):
        self.__stream = memoryview(stream)
        self.__offset = 0

        self.__data_structs = {
            "label": CString("utf8"),
            "size": Int32ul,
            "count": Int32ul,
            "frame_number": Int32ul,
            "unlabeled_marker": unlabeledMarkerStruct,
            "legacy_marker": unlabeledMarkerStruct,
            "labeled_marker": labeledMarkerStruct,
            "rigid_body": rigidBodyStruct,
        }

        self.__desc_structs = {}

    def seek(self, by: int) -> None:
        self.__offset += by

    def tell(self) -> int:
        return self.__offset

    def sizeof(self, asset_type: str, asset_count: int = 1) -> int:
        return self.__data_structs[asset_type].sizeof() * asset_count

    def parse(self, asset_type: str) -> Union[str, int, Container]:
        struct = self.__data_structs[asset_type]
        contents = struct.parse(self.__stream[self.__offset :])

        if asset_type == "label":
            self.seek(len(contents) + 1)
        else:
            self.seek(struct.sizeof())

        return contents
