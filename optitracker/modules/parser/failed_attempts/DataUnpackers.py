# TODO: Document

from construct import Struct
from DataStructures import *
from typing import Tuple, List, Dict, Union


# # # # # # # # # # # # # # # # # # # # # # #
# Parent class for asset-specific upackers  #
# # # # # # # # # # # # # # # # # # # # # # #


class dataUnpacker:
    def __init__(self, stream_version: List[int] = [4, 0, 0, 0]) -> None:
        self.stream_version = stream_version    # no intent to ever use this
        self.framedata = {}
        self.bytelength = 0

        try:
            self.structure = self.__set_structure()
        except KeyError:
            raise TypeError(f"\nNo construct.Struct() defined for {type(self)}\n")

    def __set_structure(self) -> None:
        self.structure = FRAMEDATA_STRUCTS[type(self)]

    # Shadows Construct.Struct.parse() method
    def parse(self, stream: bytes) -> None:
        self.framedata = self.structure.parse(stream)
        self.relative_offset = 
        self.structure.parse(stream)

    # Coerces data parcels into list[dict]; bundling procedure varies by asset type
    #       NOTE: Children drop terminal entries (1 = obj addr, -1 = offset)
    def data(self, stream) -> List[Dict]:
        raise NotImplementedError(
            '[dataUnpacker] child classes must define data().'
        )


# # # # # # # # # # # # # #
# Unpackers by asset type #
# # # # # # # # # # # # # #


class prefixData(dataUnpacker):
    def __init__(self, stream_version: List[int] = None) -> None:
        super().__init__(stream, stream_version)

    def data(self, stream) -> List[Dict]:
        return [dict(list(self._framedata.items())[1:-1])]


class markerSetsData(dataUnpacker):
    def __init__(self, stream_version: List[int] = [4,0,0,0]) -> None:
        super().__init__(stream_version)

    def data(self, stream: bytes) -> List[Dict]:
        return [
            dict(list(marker.items())[1:])
            for marker_set in self._framedata.children
            for marker in marker_set.children
        ]


class labeledMarkerSetData(dataUnpacker):
    def __init__(self, stream_version: List[int] = None) -> None:
        super().__init__(stream, stream_version)

    def data(self, stream) -> List[Dict]:
        return [
            dict(list(labeledMarker.items())[1:])
            for labeledMarker in self._framedata.children
        ]


class legacyMarkerSetData(dataUnpacker):
    def __init__(self, stream_version: List[int] = None) -> None:
        super().__init__(stream, stream_version)

    def data(self, stream) -> List[Dict]:
        return [
            dict(list(legacyMarker.items())[1:])
            for legacyMarker in self._framedata.children
        ]


class rigidBodiesData(dataUnpacker):
    def __init__(self, stream_version: List[int] = None) -> None:
        super().__init__(stream, stream_version)

    def data(self, stream) -> List[Dict]:

        return [
            dict(list(rigidBody.items())[1:])
            for rigidBody in self._framedata.children
        ]


class skeletonsData(dataUnpacker):
    def __init__(self, stream_version: List[int] = None) -> None:
        super().__init__(stream, stream_version)

    def data(self, stream) -> List[Dict]:
        return [
            dict(list(rigidBody.items())[1:])
            for skeleton in self._framedata.children
            for rigidBody in skeleton.children
        ]


class assetsData(dataUnpacker):
    def __init__(self, stream_version: List[int] = None) -> None:
        super().__init__(stream, stream_version)

    def data(self, asset_type) -> List[Dict]:
        if asset_type == 'AssetRigidBodies':
            return [
                dict(list(assetRigidBody.items())[1:])
                for assetRigidBody in self._framedata.rigid_body_children
            ]
            # for asset in self._framedata.children
            # for assetRigidBody in asset.rigid_bodies]

        elif asset_type == 'AssetMarkers':
            return [
                dict(list(assetMarker.items())[1:])
                for assetMarker in self._framedata.marker_children
            ]
            # for asset in self._framedata.children
            # for assetMarker in asset.markers]
        else:
            raise ValueError(
                f"assetData.export() | asset_type must be 'AssetRigidBodies' or 'AssetMarkers'; type supplied: {asset_type}"
            )


class forcePlatesData(dataUnpacker):
    def __init__(self, stream_version: List[int] = None) -> None:
        super().__init__(stream, stream_version)

    def data(self, stream) -> List[Dict]:
        return [
            dict(list(frame.items())[1:])
            for forcePlate in self._framedata.children
            for channel in forcePlate.children
            for frame in channel.children
        ]


class devicesData(dataUnpacker):
    def __init__(self, stream_version: List[int] = None) -> None:
        super().__init__(stream, stream_version)

    def data(self, stream) -> List[Dict]:
        return [
            dict(list(frame.items())[1:])
            for device in self._framedata.children
            for channel in device.children
            for frame in channel.children
        ]


class suffixData(dataUnpacker):
    def __init__(self, stream_version: List[int] = None) -> None:
        super().__init__(stream, stream_version)

    def data(self, stream) -> List[Dict]:
        return [dict(list(self._framedata.items())[1:-1])]


# Unpacker class type used to select the correct structure
FRAMEDATA_STRUCTS = {
    prefixData: dataStruct_Prefix,
    markerSetsData: dataStruct_MarkerSets,
    labeledMarkerSetData: dataStruct_LabeledMarkerSet,
    legacyMarkerSetData: dataStruct_LegacyMarkerSet,
    rigidBodiesData: dataStruct_RigidBodies,
    skeletonsData: dataStruct_Skeletons,
    assetsData: dataStruct_Assets,
    forcePlatesData: dataStruct_ForcePlates,
    devicesData: dataStruct_Devices,
    suffixData: dataStruct_Suffix,
}

# # # # # # # # # # # # #
# Frame data container  #
# # # # # # # # # # # # #


class frameData:
    def __init__(self) -> None:

        # Aggregate Frame Data
        self._framedata = {
            'Prefix': [],
            'MarkerSets': [],
            'LabeledMarkerSets': [],
            'LegacyMarkerSets': [],
            'RigidBodies': [],
            'Skeletons': [],
            'AssetRigidBodies': [],
            'AssetMarkers': [],
            'ForcePlates': [],
            'Devices': [],
            'Suffix': [],
        }

    def log(self, asset_type: str, asset_frame_data: List[Dict]) -> None:
        self._framedata[asset_type].append(asset_frame_data)

    # Log frame data for a given asset type
    def __validate_export_arg(
        self, arg: Union[Tuple[str, ...], str], name: str
    ) -> Tuple[str, ...]:
        if isinstance(arg, str):
            return (arg,)
        elif isinstance(arg, tuple) and all(isinstance(i, str) for i in arg):
            return arg
        else:
            raise TypeError(
                f'frameData.export() | {name}: expected str or tuple thereof, got {type(arg)}'
            )

    # Export frame data for desired asset types; also allows for omission
    def export(
        self,
        include: Union[Tuple[str, ...], str] = None,
        exclude: Union[Tuple[str, ...], str] = None,
    ) -> Dict[str, Tuple[Dict, ...]]:
        # include = self.__validate_export_arg(include, "include")

        # if exclude is not None:
        #     exclude = self.__validate_export_arg(exclude, "exclude")
        #     return {k: v for k, v in self._framedata.items()
        #             if k in include and k not in exclude}

        # return {k: v for k, v in self._framedata.items() if k in include}

        return self._framedata
