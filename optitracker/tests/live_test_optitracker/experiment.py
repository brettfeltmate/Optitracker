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
from klibs.KLUserInterface import ui_request, any_key, smart_sleep
from klibs.KLUtilities import pump
from klibs.KLBoundary import BoundaryInspector, CircleBoundary

from klibs.KLTime import CountDown

# local modules
from natnetclient_rough import NatNetClient  # type: ignore[import]
from OptiTracker import OptiTracker  # type: ignore[import]

RED = "red"
BLUE = "blue"
GREEN = "green"


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

        self.data_file = os.getcwd() + "\\test_optitracker_data_run_{}.csv".format(P.p_id)

        self.ot = OptiTracker(marker_count=3)
        self.ot.window_size = 10
        self.ot.data_dir = self.data_file

        self.nnc = NatNetClient()
        self.nnc.markers_listener = self.marker_set_listener

        self.fills = {
            RED: [255, 000, 000, 255],
            GREEN: [000, 255, 000, 255],
            BLUE: [000, 000, 255, 255]
        }

        self.locs = {
            RED: (P.screen_c[0] - OFFSET, P.screen_c[1]),  # type: ignore
            GREEN: P.screen_c,  # type: ignore
            BLUE: (P.screen_c[0] + OFFSET, P.screen_c[1]),  # type: ignore
        }

        self.placeholders = {
            RED: kld.Annulus(BOUNDARY_DIAM, BRIMWIDTH, fill=self.fills[RED]),
            GREEN: kld.Annulus(BOUNDARY_DIAM, BRIMWIDTH, fill=self.fills[GREEN]),
            BLUE: kld.Annulus(BOUNDARY_DIAM, BRIMWIDTH, fill=self.fills[BLUE]),
        }

        self.cursor = kld.Circle(BOUNDARY_DIAM // 2, fill=[255, 255, 255, 255])

        self.red_boundary = CircleBoundary(RED, self.locs[RED], BOUNDARY_DIAM)
        self.green_boundary = CircleBoundary(GREEN, self.locs[GREEN], BOUNDARY_DIAM)
        self.blue_boundary = CircleBoundary(BLUE, self.locs[BLUE], BOUNDARY_DIAM)

        self.bi = BoundaryInspector(
            [self.red_boundary, self.blue_boundary, self.green_boundary]
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

        smart_sleep(100)

    def trial(self):  # type: ignore

        counter = CountDown(60)
        while counter.counting():
            q = pump()
            _ = ui_request(queue=q)

            fill()

            for key in self.locs.keys():
                blit(self.placeholders[key], location=self.locs[key], registration=5)

            cursor_pos = self.ot.position()
            # print("Cursor pos: {}".format(cursor_pos))

            pos = [int(cursor_pos["pos_x"][0]), int(cursor_pos["pos_z"][0])]

            cursor_vel = self.ot.velocity()

            which_bound = self.bi.which_boundary(pos)

            if which_bound is not None:
                self.cursor.fill = self.fills[which_bound]

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

        return {"block_num": P.block_number, "trial_num": P.trial_number}

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
        if not os.path.exists(self.data_file):
            with open(self.data_file, "a", newline="") as file:
                writer = DictWriter(file, fieldnames=marker_set["markers"][0].keys())
                writer.writeheader()

        with open(self.data_file, "a", newline="") as file:
            writer = DictWriter(file, fieldnames=marker_set["markers"][0].keys())
            for marker in marker_set.get("markers", None):
                writer.writerow(marker)
