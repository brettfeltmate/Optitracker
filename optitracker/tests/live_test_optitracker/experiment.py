# -*- coding: utf-8 -*-

# external libs
import os
from pathlib import Path
from csv import DictWriter

import klibs
from klibs import P
from klibs.KLCommunication import message
from klibs.KLGraphics import KLDraw as kld
from klibs.KLGraphics import blit, fill, flip
from klibs.KLUserInterface import ui_request, any_key
from klibs.KLUtilities import pump
from klibs.KLBoundary import BoundaryInspector, CircleBoundary

# local modules
from natnetclient_rough import NatNetClient  # type: ignore[import]
from OptiTracker import OptiTracker  # type: ignore[import]

LEFT = "left"
RIGHT = "right"
CENTER = "center"

RED = [255, 0, 0, 255]
BLUE = [0, 0, 255, 255]
GREEN = [0, 255, 0, 255]


class live_test_optitracker(klibs.Experiment):

    def setup(self):
        # self.dir = {}
        # self.dir["root"] = Path('.')
        # self.dir["optidata"] = self.dir["root"] / "optitracker_data"
        # self.


        self.px_mm = int(P.ppi / 2.54) // 10  # pixels per mm
        OFFSET = self.px_mm * 100
        BOUNDARY_DIAM = self.px_mm * 50
        BRIMWIDTH = self.px_mm * 5

        # self.path = Path('.')
        # self.data_file = self.path / "test_optitracker_data_run.csv"

        self.data_file = os.getcwd() + "/test_optitracker_data_run_{}.csv".format(P.p_id)
        print(self.data_file)

        self.ot = OptiTracker(marker_count=3)
        self.ot.window_size = 2
        self.ot.data_dir = self.data_file

        self.nnc = NatNetClient()
        self.nnc.markers_listener = self.marker_set_listener

        self.locs = {
            LEFT: (P.screen_c[0] - OFFSET, P.screen_c[1]),  # type: ignore
            CENTER: P.screen_c,  # type: ignore
            RIGHT: (P.screen_c[0] + OFFSET, P.screen_c[1]),  # type: ignore
        }

        self.placeholders = {
            LEFT: kld.Annulus(BOUNDARY_DIAM, BRIMWIDTH, fill=RED),
            CENTER: kld.Annulus(BOUNDARY_DIAM, BRIMWIDTH, fill=GREEN),
            RIGHT: kld.Annulus(BOUNDARY_DIAM, BRIMWIDTH, fill=BLUE),
        }

        self.cursor = kld.Circle(BOUNDARY_DIAM // 2, fill=[255, 255, 255, 255])

        self.left_boundary = CircleBoundary(LEFT, self.locs[LEFT], BOUNDARY_DIAM)
        self.right_boundary = CircleBoundary(RIGHT, self.locs[LEFT], BOUNDARY_DIAM)
        self.center_boundary = CircleBoundary(CENTER, self.locs[LEFT], BOUNDARY_DIAM)

        self.bi = BoundaryInspector(
            [self.left_boundary, self.right_boundary, self.center_boundary]
        )

    def block(self):
        pass

    def trial_prep(self):

        fill()
        message(
            "Press any key to start the trial, & cmd-q to quit",
            location=P.screen_c,
            registration=5,
            blit_txt=True,
        )
        flip()

        any_key()

        fill()
        flip()
        self.nnc.startup()

    def trial(self):  # type: ignore

        while True:
            q = pump()
            _ = ui_request(queue=q)

            fill()

            for loc in self.locs.keys():
                blit(self.placeholders[loc], location=self.locs[loc], registration=5)

            cursor_pos = self.ot.position()
            cursor_pos["pos_x"] = cursor_pos["pos_x"] * 1000
            cursor_pos["pos_y"] = cursor_pos["pos_y"] * 1000
            cursor_pos["pos_z"] = cursor_pos["pos_z"] * 1000

            pos = [cursor_pos["pos_x"][0], cursor_pos["pos_z"][0]]
            pos = [int(p * self.px_mm) for p in pos]

            cursor_vel = self.ot.velocity() * 100

            which_bound = self.bi.which_boundary(pos)

            if which_bound is not None:
                self.cursor.fill = self.placeholders[which_bound].fill

            else:
                self.cursor.fill = [255, 255, 255, 255]

            blit(self.cursor, location=pos, registration=5)

            msg = f"X: {pos[0]}\n"
            msg += f"Z: {pos[1]}\n"
            msg += f"Vel: {cursor_vel}\n"
            msg += f"Bound: {which_bound}"

            message(
                text=msg,
                location=[P.screen_c[0] // 10, P.screen_c[1] // 10],  # type: ignore
                registration=7,
                blit_txt=True,
            )

            flip()

        # return {"block_num": P.block_number, "trial_num": P.trial_number}

    def trial_clean_up(self):
        pass

    def clean_up(self):
        pass

    def marker_set_listener(self, marker_set: dict) -> None:
        """Write marker set data to CSV file.

        Args:
            marker_set (dict): Dictionary containing marker data to be written.
                Expected format: {'markers': [{'key1': val1, ...}, ...]}
        """
        # print(marker_set["markers"][0].keys())
        with open(self.data_file, "a", newline="") as file:
            writer = DictWriter(file, fieldnames=marker_set["markers"][0].keys())
            if not os.path.exists(self.data_file):
                writer.writeheader()

            for marker in marker_set.get("markers", None):
                writer.writerow(marker)
