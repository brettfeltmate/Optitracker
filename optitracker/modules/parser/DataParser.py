# type: ignore

from optitracker.modules.parser.ParsingStructures import structs

from typing import Container


class DataParser(object):
    def __init__(self, stream: bytes):
        self.__stream = memoryview(stream)
        self.__offset = 0

    def seek(self, by: int) -> None:
        self.__offset += by

    def tell(self) -> int:
        return self.__offset

    def size(self, asset: str, asset_count: int = 1) -> int:
        return structs[asset].sizeof() * asset_count

    def bytelen(self) -> tuple[int, int]:
        bytelen = structs["size"].parse(self.__stream[self.__offset :])
        my_size = self.size("size")
        self.seek(my_size)

        return (bytelen, my_size)

    def frame_number(self) -> tuple[int, int]:
        num = structs["frame_number"].parse(self.__stream[self.__offset :])
        my_size = self.size("frame_number")
        
        return (num, my_size)

    def count(self) -> tuple[int, int]:
        count = structs["count"].parse(self.__stream[self.__offset :])
        self.seek(structs["count"].sizeof())
        return (count, self.size("count"))

    def label(self) -> tuple[str, int]:
        label = structs["label"].parse(self.__stream[self.__offset :])
        my_size = len(label) + 1
        self.seek(my_size)
        return (label, my_size)

    def struct(self, asset: str) -> tuple[Container, int]:
        if asset == "count":
            raise ValueError("Use count() method to parse count struct")

        if asset == "label":
            raise ValueError("Use label() method to parse label struct")

        struct = structs[asset]
        contents = struct.parse(self.__stream[self.__offset :])
        my_size = struct.sizeof()

        self.seek(my_size)

        return (contents, my_size)
