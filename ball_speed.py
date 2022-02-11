import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from enum import Enum

BALL_MARK_EXIST_THRESHOLD = 125
BALL_MARK_STEADY_FRAMES = 15


class BallPos(Enum):
    START = 1
    END = 2
    MIDDLE = 3
    EMPTY = 4


class BallSpeed:
    def __init__(self, fps, debug=False):
        self.debug = debug
        self.fps = fps
        self.window = int(fps)
        self.frames_pos = []
        self.speeds = []  # RPS
        self.counts = []
        self.frame_idx = -1

    def debug_plot_speed(self):
        plt.plot(self.speeds)
        plt.ylabel('RPC')
        plt.ylabel('frame')
        plt.show()

    def process(self, img):
        self.frame_idx += 1
        ball = self.__get_ball(img)
        if ball is None:
            return self.speeds[-1] if len(self.speeds) else 0

        ball_pos = self.__get_ball_position(ball)
        self.frames_pos.append(ball_pos)
        self.__calc_changes()
        return self.speeds[-1]

    def __pos_to_count(self):
        try:
            if len(self.frames_pos) > BALL_MARK_STEADY_FRAMES:
                lasts = set(self.frames_pos[len(self.frames_pos) - BALL_MARK_STEADY_FRAMES:])
                if len(lasts) == 1:
                    return 0

            last_pos = self.frames_pos[-2]
            curr_pos = self.frames_pos[-1]

            if curr_pos == BallPos.EMPTY:
                return 0
            if last_pos == BallPos.START and (
                    curr_pos == BallPos.MIDDLE or curr_pos == BallPos.END):
                return 0
            return 1

        except IndexError:  # on first frame
            self.counts.append(0)
            return 0

    def __calc_changes(self):
        self.counts.append(self.__pos_to_count())
        self.speeds.append(sum(self.counts))
        if len(self.counts) >= self.window:
            self.counts = self.counts[1:]

    def __get_ball(self, im):
        """
        :param im: image from camera
        :return: ball image in gray scale - sub image or None if not found it
        """
        gim = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gim, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=40, minRadius=30, maxRadius=50)
        if circles is None:
            if self.debug:
                print('didnt found circle')
            return None
        circles = np.uint16(np.around(circles))[0, :]
        x, y, r = circles[0]
        cv2.circle(im, (x, y), r, (0, 0, 0), 3)
        ball = gim[y - r:y + r, x - r:x + r]

        if self.debug:
            print('circles found:', len(circles))
            print(x, y, r)
            # cv2.imwrite('frames/big/{}.jpg'.format(self.frame), ball)
        return ball

    def __get_ball_position(self, ball) -> BallPos:
        """
        :param ball: gray image
        :return: position: BallPos
        """
        radius = int(ball.shape[0] / 2)
        # Get binary marks of the ball
        tone_map = (np.array(range(256)) < BALL_MARK_EXIST_THRESHOLD) * 255
        bin_marks = tone_map[ball.astype(int)].astype(int)

        if self.debug:
            plt.imshow(bin_marks, cmap='gray')
            plt.show()

        # Determine the ball's marks position
        pos_dict = {'top': 0, 'bottom': 0}
        mid_r, mid_c = radius, radius
        for r in range(ball.shape[0]):
            for c in range(ball.shape[1]):
                dist_from_middle = math.sqrt((r - mid_r) ** 2 + (c - mid_c) ** 2)
                if dist_from_middle >= 0.9 * radius:  # Ignore outside of circle and edges
                    continue
                if bin_marks[r, c] == 255:
                    pos_key = 'top' if r < radius else 'bottom'
                    pos_dict[pos_key] += 1

        circle_to_sq_ratio = 0.78  # pi / 4
        pos_min_count = int(ball.size * circle_to_sq_ratio * 0.0025)
        if pos_dict['top'] >= pos_min_count > pos_dict['bottom']:
            return BallPos.START
        elif pos_dict['bottom'] >= pos_min_count > pos_dict['top']:
            return BallPos.END
        elif pos_dict['bottom'] < pos_min_count > pos_dict['top']:
            return BallPos.EMPTY
        elif pos_dict['bottom'] >= pos_min_count <= pos_dict['top']:
            return BallPos.MIDDLE
