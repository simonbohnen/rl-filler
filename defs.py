from pygame.locals import K_0, K_1, K_2, K_3, K_4, K_5
from typing import List, Tuple, Set

WIDTH = 8
HEIGHT = 7
CELLSIZE = 50

BLACK = (60, 60, 60)
YELLOW = (251, 224, 79)
PURPLE = (102, 79, 158)
GREEN = (166, 203, 97)
BLUE = (82, 166, 237)
RED = (229, 72, 88)
COLORS = [BLACK, YELLOW, PURPLE, GREEN, BLUE, RED]
COLOR_KEYS = [K_0, K_1, K_2, K_3, K_4, K_5]
COLOR_NAMES = ["black", "yellow", "purple", "green", "blue", "red"]

Point = Tuple[int, int]
Cellset = Set[Point]
Field = List[List[bool]]
Board = List[List[int]]
