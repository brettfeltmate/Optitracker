import unittest
from MotiveStreamParser import MotiveStreamParser
import sys
import os
from pprint import pprint


class TestMotiveStreamParser(unittest.TestCase):
    def setup(self):
        # This runs before each test
        pass

    def test_parser_functionality(self):
        """Test specific parser functionality
        MoCap frame unpacking sequence
            1. prefix
            2. marker sets
            3. legacy markers
            4. rigid bodies
            5. skeletons
            6. assets
            7. labeled marker sets
            8. force plates
            9. devices
            10. suffix
        """
        self.data_file = None
        if len(sys.argv) > 1:
            self.data_file = sys.argv[1]
            if not os.path.exists(self.data_file):
                raise FileNotFoundError(f"Data file not found: {self.data_file}")

        with open(self.data_file, "rb") as f:  # type: ignore
            self.data = f.read()
        self.parser = MotiveStreamParser(self.data)

        prefix = self.parser.parse("frame_number")

        print("Prefix:")
        pprint(prefix)
        print(f"Pos: {self.parser.tell()}")

        n_marker_sets = self.parser.parse("count")

        print("Sets:")
        pprint(n_marker_sets)
        print(f"Pos: {self.parser.tell()}")

        size = self.parser.parse("size")

        print("Size:")
        pprint(size)
        print(f"Pos: {self.parser.tell()}")

        for i in range(n_marker_sets):  # type: ignore
            print(f"Set {i}")
            pprint(f"Marker Set {i + 1}")

            label = self.parser.parse("label")

            print("Label:")
            pprint(label)
            print(f"Pos: {self.parser.tell()}")

            n_markers_in_set = self.parser.parse("count")

            print("Num markers in set:")
            pprint(n_markers_in_set)
            print(f"Pos: {self.parser.tell()}")

            marker_set = {"label": label, "markers": []}

            for j in range(n_markers_in_set):  # type: ignore
                print(f"Marker {j + 1}")

                marker = self.parser.parse("unlabeled_marker")

                print("Marker:")
                pprint(marker)
                print(f"Pos: {self.parser.tell()}")

                marker_set["markers"].append(marker)

            print(f"Marker Set {i + 1}")
            pprint(marker_set)
            print(f"Pos: {self.parser.tell()}")

        n_legacy_markers = self.parser.parse("count")

        print("Legacy Marker:")
        pprint(n_legacy_markers)
        print(f"Pos: {self.parser.tell()}")

        size = self.parser.parse("size")

        print("Size:")
        pprint(size)
        print(f"Pos: {self.parser.tell()}")

        legacy_markers = []

        for i in range(n_legacy_markers):  # type: ignore
            print(f"Legacy Marker {i + 1}")

            # label = self.parser.parse("label")
            #
            # print("Label:")
            # pprint(label)
            # print(f"Pos: {self.parser.tell()}")
            legacy_marker = self.parser.parse("legacy_marker")

            print("Legacy Marker:")
            pprint(legacy_marker)
            print(f"Pos: {self.parser.tell()}")

            legacy_markers.append(legacy_marker)

            print("Legacy Marker")
            pprint(legacy_markers)
            print(f"Pos: {self.parser.tell()}")

        n_rigid_bodies = self.parser.parse("count")

        print("RB Count:")
        pprint(n_rigid_bodies)
        print(f"Pos: {self.parser.tell()}")

        size = self.parser.parse("size")

        print("Size:")
        pprint(size)
        print(f"Pos: {self.parser.tell()}")

        for i in range(n_rigid_bodies):
            print(f"RB num: {i}")

            rigid_body = self.parser.parse("rigid_body")

            print("Rigid Body:")
            pprint(rigid_body)
            print(f"Pos: {self.parser.tell()}")

        n_skeletons = self.parser.parse("count")

        print("Skeleton Count:")
        pprint(n_skeletons)
        print(f"Pos: {self.parser.tell()}")

        size = self.parser.parse("size")

        print("Size:")
        pprint(size)
        print(f"Pos: {self.parser.tell()}")

        for i in range(n_skeletons):
            print(f"Skeleton num: {i}")

            skeleton_id = self.parser.parse("count")

            print("Skeleton ID:")
            pprint(skeleton_id)
            print(f"Pos: {self.parser.tell()}")

            n_rigid_bodies = self.parser.parse("count")

            print("RB Count:")
            pprint(n_rigid_bodies)
            print(f"Pos: {self.parser.tell()}")

            for j in range(n_rigid_bodies):
                print(f"RB num: {j}")

                rigid_body = self.parser.parse("rigid_body")

                print("Rigid Body:")
                pprint(rigid_body)
                print(f"Pos: {self.parser.tell()}")


if __name__ == "__main__":
    # This allows you to pass a file path as an argument
    # Usage: python test_parser.py path/to/your/data/file
    unittest.main(argv=["first-arg-is-ignored"])
    input("Press Enter to exit...")
