import random
from defs import *
import pygame


def set_color(board, mousepos, color):
    x = mousepos[0] // CELLSIZE
    y = mousepos[1] // CELLSIZE
    if x >= WIDTH or y >= HEIGHT:
        return
    board[x][y] = color


def get_color(board, mousepos):
    x = mousepos[0] // CELLSIZE
    y = mousepos[1] // CELLSIZE
    if x >= WIDTH or y >= HEIGHT:
        return
    print("Clicked on cell: %d, %d" % (x, y))
    return board[x][y]


def draw_board(board, surface):
    for x in range(WIDTH):
        for y in range(HEIGHT):
            color = board[x][y]
            if not color == -1:
                pygame.draw.rect(surface, COLORS[color],
                                 pygame.Rect(CELLSIZE * x, CELLSIZE * y, CELLSIZE, CELLSIZE))


def random_color_exluding(exclude=None):
    if exclude is None:
        exclude = []
    color = random.randrange(len(COLORS))
    while color in exclude:
        color = random.randrange(len(COLORS))
    return color


def get_random_regular_board() -> Board:
    """
    returns a new random regular board. regular means that no two direct neighbors have the same color.
    :return:
    """
    # Fill top left cell
    board = [[0 for _ in range(HEIGHT)] for _ in range(WIDTH)]
    board[0][0] = random_color_exluding()
    # Fill left column
    for y in range(0, HEIGHT - 1):
        board[0][y+1] = random_color_exluding([board[0][y]])
    # Fill top row
    for x in range(0, WIDTH - 1):
        board[x+1][0] = random_color_exluding([board[x][0]])
    # Fill rest
    for x in range(0, WIDTH - 1):
        for y in range(0, HEIGHT - 1):
            board[x+1][y+1] = random_color_exluding([board[x][y + 1], board[x + 1][y]])
    return board


def str_board(board):
    result = ""
    for row in list(map(list, zip(*board))):
        result += " ".join(map(str, row)) + "\n"
    return result


def board_like_on_screen(board):
    return list(map(list, zip(*board)))


def print_colors():
    for i in range(len(COLORS)):
        print("Color %d: %s" % (i, COLOR_NAMES[i]))
