import random
from typing import List, Tuple, Set

import pygame
from pygame.locals import QUIT, KEYDOWN, K_RETURN, MOUSEBUTTONDOWN
import numpy as np

WIDTH = 8
HEIGHT = 7
CELLSIZE = 50

BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
PURPLE = (255, 0, 255)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
COLORS = (BLACK, YELLOW, PURPLE, GREEN, BLUE, RED)

'''
Data:
Board: 2D list of colors
adjacents: list of lists of positions of cells adjacent to conquered part indexed by color
checked: indicates whether cell has been considered for adjacency
'''


class PlayerState:
    def __init__(self, board, isplayer1):
        """

        :param board:
        :param isplayer1:
        """
        self.adjacents: List[Set[Tuple[int, int]]] = [set() for _ in range(6)]
        self.score = 1
        self.owned = set()
        if isplayer1:
            self.owned.add((0, HEIGHT - 1))
            self.adjacents[board[0][HEIGHT - 2]].add((0, HEIGHT - 2))
            self.adjacents[board[1][HEIGHT - 1]].add((1, HEIGHT - 1))
        else:
            self.owned.add((WIDTH - 1, 0))
            self.adjacents[board[WIDTH - 1][1]].add((WIDTH - 1, 1))
            self.adjacents[board[WIDTH - 2][0]].add((WIDTH - 2, 0))

    def move(self, board, color, is_available):
        for x, y in self.owned:
            board[x][y] = color

        self.owned |= self.adjacents[color]
        self.score += len(self.adjacents[color])
        new_owned = self.adjacents[color]
        for x, y, in new_owned:
            is_available[x][y] = False
        self.adjacents[color] = set()

        self.update_adjacents(board, new_owned, is_available)
        return new_owned

    def update_adjacents(self, board, new_owned, is_available):
        to_add = []
        for x, y in new_owned:
            to_add += [(x+1, y), (x, y+1), (x-1, y), (x, y-1)]

        to_add_set = set([(x, y) for x, y in to_add if 0 <= x < WIDTH and 0 <= y < HEIGHT and is_available[x][y]])
        for x, y in to_add_set:
            self.adjacents[board[x][y]].add((x, y))

    def remove_new_owned_from_adjacents(self, new_owned):
        for c in range(len(COLORS)):
            self.adjacents[c] -= new_owned


class FillerState:
    def __init__(self, board, starting_player):
        self.board = board
        # 1 is player in bottom left corner (0, HEIGHT - 1), 2 is player in top right corner (WIDTH - 1, 0)
        self.player = starting_player
        self.player1State = PlayerState(self.board, True)
        self.player2State = PlayerState(self.board, False)
        self.available = [[True for _ in range(HEIGHT)] for _ in range(WIDTH)]
        self.available[0][HEIGHT - 1] = False
        self.available[WIDTH - 1][0] = False
        self.is_final_state = False

    def move(self, color):
        if color == self.board[WIDTH - 1][0] or color == self.board[0][HEIGHT - 1]:
            print("Can't choose color %d" % color)
            return False

        if self.player == 1:
            # Bottom left player
            new_owned = self.player1State.move(self.board, color, self.available)
            self.player2State.remove_new_owned_from_adjacents(new_owned)
            self.player = 2
        else:
            # Top right player
            new_owned = self.player2State.move(self.board, color, self.available)
            self.player1State.remove_new_owned_from_adjacents(new_owned)
            self.player = 1
        if self.player1State.score + self.player2State.score == WIDTH * HEIGHT:
            self.is_final_state = True
        return True

    def do_standard_move(self):
        if self.player == 1:
            current_state = self.player1State
        else:
            current_state = self.player2State

        player1_color = self.board[0][HEIGHT-1]
        player2_color = self.board[WIDTH-1][0]
        max_adjacents = 0
        greedy_color = random_color_exluding([player1_color, player2_color])
        adjacent_colors = set()

        valid_colors = (c for c in range(len(COLORS)) if c != player1_color and c != player2_color)
        for color in valid_colors:
            adjacent_count = len(current_state.adjacents[color])
            if adjacent_count > 0:
                adjacent_colors.add(color)
            if adjacent_count > max_adjacents:
                max_adjacents = adjacent_count
                greedy_color = color

        ran = random.random()
        adjacent_colors.discard(greedy_color)
        if ran < 0.7 or not adjacent_colors:
            print("Chose greedy color (could be random)")
            self.move(greedy_color)
        else:
            print("Chose random adjacent color")
            self.move(random.choice(list(adjacent_colors)))


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
    return board[x][y]


def draw_board(board, surface):
    for x in range(WIDTH):
        for y in range(HEIGHT):
            color = board[x][y]
            if not color == -1:
                pygame.draw.rect(surface, COLORS[board[x][y]],
                                 pygame.Rect(CELLSIZE * x, CELLSIZE * y, CELLSIZE, CELLSIZE))


def random_color_exluding(exclude=None):
    if exclude is None:
        exclude = []
    color = random.randrange(len(COLORS))
    while color in exclude:
        color = random.randrange(len(COLORS))
    return color


def get_random_regular_board():
    # Fill top left cell
    board = np.zeros((8, 7), dtype=np.int32)
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


# Board [x][y], transpose at \


def main():
    # Initialize Everything
    pygame.init()
    screen = pygame.display.set_mode((400, 350))
    pygame.display.set_caption('Filler')

    # Create The Background
    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((255, 255, 255))  # Unnecessary

    # Prepare Game Objects
    clock = pygame.time.Clock()

    running = True
    initing = False  # Indicates whether we want to enter a board by hand
    color = 0
    board = np.full((8, 7), -1, dtype=np.int32) if initing else get_random_regular_board()
    state = None if initing else FillerState(board, 1)

    draw_board(board, background)
    print(str_board(board))

    while running:
        clock.tick(60)

        # Handle Input Events
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif initing and event.type == KEYDOWN and event.key == K_RETURN:
                color += 1
                if color == len(COLORS):
                    initing = False
                    state = FillerState(board, 1)
            elif event.type == MOUSEBUTTONDOWN:
                if initing:
                    set_color(board, pygame.mouse.get_pos(), color)
                else:
                    if state.move(get_color(state.board, pygame.mouse.get_pos())) and not state.is_final_state:
                        state.do_standard_move()
                    if state.is_final_state:
                        print("Finished game. Player 1 scored %d, player 2 scored %d." % (state.player1State.score, state.player2State.score))
                        board = get_random_regular_board()
                        state = FillerState(board, 1)

                draw_board(board, background)
        # Draw Everything
        screen.blit(background, (0, 0))
        pygame.display.flip()

    pygame.quit()


# this calls the 'main' function when this script is executed
if __name__ == '__main__':
    main()
