from defs import *
from playerstate import PlayerState
from helpers import random_color_exluding
import random
import sys
import copy

neginf = sys.float_info.min
inf = sys.float_info.max

class FillerState:
    def __init__(self, board: Board, starting_player: int, quiet: bool = True):
        """
        Creates a new FillerState.
        :param board: the board.
        :param starting_player: the starting player
        :param quiet: whether printing should be quiet for this instance.
        """
        self.board = board
        # 1 is player in bottom left corner (0, HEIGHT - 1), 2 is player in top right corner (WIDTH - 1, 0)
        self.player = starting_player
        self.player1State = PlayerState(self.board, True)
        self.player2State = PlayerState(self.board, False)
        self.is_available = [[True for _ in range(HEIGHT)] for _ in range(WIDTH)]
        self.is_available[0][HEIGHT - 1] = False
        self.is_available[WIDTH - 1][0] = False
        self.is_final_state = False
        self.move_count = 0
        self.quiet = quiet
        self.deadquiet = False
        self.last_move_illegal = False

    def move(self, color: int) -> int:
        """
        perform the move indicated by `color` for the current player. Current player switches.
        :param color: the color to move with.
        :return: the number of newly owned cells including enclaves.
        """
        self.move_count += 1
        self.last_move_illegal = False

        # Determine whether move was illegal
        player1_color = self.board[0][HEIGHT - 1]
        player2_color = self.board[WIDTH - 1][0]
        if color == player1_color or color == player2_color:
            oldcolor = color
            color = random_color_exluding([player1_color, player2_color])
            if not self.quiet:
                print("INFO: Illegal move by player %d: %s. Chose %s randomly."
                      % (self.player, COLOR_NAMES[oldcolor], COLOR_NAMES[color]))
            self.last_move_illegal = True

        if not self.deadquiet:
            print(f"MOVE: Player {self.player} chose {COLOR_NAMES[color]}.")

        if self.player == 1:
            # Bottom left player
            old_score = self.player1State.score
            new_owned = self.player1State.move(self.board, color, self.is_available,
                                               self.player2State.owned)
            self.player2State.remove_new_owned_from_adjacents(new_owned)
            self.player = 2
            score_diff = self.player1State.score - old_score
            if not self.quiet:
                print("INFO: Player 1 scored %d points. Score is now %d." % (score_diff, self.player1State.score))
        else:
            # Top right player
            old_score = self.player2State.score
            new_owned = self.player2State.move(self.board, color, self.is_available, self.player1State.owned)
            self.player1State.remove_new_owned_from_adjacents(new_owned)
            self.player = 1
            score_diff = self.player2State.score - old_score
            if not self.quiet:
                print("INFO: Player 2 scored %d points. Score is now %d." % (score_diff, self.player2State.score))

        area = WIDTH * HEIGHT
        if self.player1State.score > area / 2 or \
                self.player2State.score > area / 2 or \
                self.player1State.score + self.player2State.score == area:
            self.is_final_state = True
        return score_diff

    def do_standard_move(self):
        """
        performs a standard move, choosing the color with the most adjacents with a certain probability.
        """
        if self.player == 1:
            current_state = self.player1State
        else:
            current_state = self.player2State

        player1_color = self.board[0][HEIGHT-1]
        player2_color = self.board[WIDTH-1][0]
        max_adjacents = 0
        greedy_color = random_color_exluding([player1_color, player2_color])
        adjacent_colors = set()

        valid_colors = (c for c in range(COLORCOUNT) if c != player1_color and c != player2_color)
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
            if not self.quiet:
                print("INFO: Chose greedy color (could be random)")
            self.move(greedy_color)
        else:
            if not self.quiet:
                print("INFO: Chose random adjacent color")
            self.move(random.choice(list(adjacent_colors)))

    def do_alpha_move(self):
        quietcopy = copy.deepcopy(self)
        quietcopy.quiet = True
        quietcopy.deadquiet = True
        (bestcolor, sc) = quietcopy.a_star_search(7)
        self.move(bestcolor)

    def a_star_search(self, depth):
        return self.a_star_search_rec(depth, neginf, inf)

    def a_star_search_rec(self, depth, alpha, beta):
        """
        A star search.
        Player 1 is maximizing, player 2 is minimizing.
        :param depth: the depth.
        :param alpha:
        :param beta:
        """
        if depth == 0 or self.is_final_state:
            return -1, self.heuristic()

        if self.player == 1:
            value = neginf
            state = self.player1State
            colors = [color for color in range(COLORCOUNT) if len(state.adjacents[color]) != 0]
            try:
                colors.remove(self.board[WIDTH - 1][0])
            except ValueError:
                pass
            colors.sort(key=lambda c: len(state.adjacents[c]))

            bestcolor = 0
            for color in colors:
                # Copy state
                newstate = copy.deepcopy(self)
                newstate.move(color)
                (bc, newval) = newstate.a_star_search_rec(depth - 1, alpha, beta)
                if newval > value:
                    value = newval
                    bestcolor = color
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return bestcolor, value
        else:
            value = inf
            state = self.player2State
            colors = [color for color in range(COLORCOUNT) if len(state.adjacents[color]) != 0]
            try:
                colors.remove(self.board[0][HEIGHT - 1])
            except ValueError:
                pass
            colors.sort(key=lambda c: len(state.adjacents[c]))

            bestcolor = 0
            for color in colors:
                # Copy state
                newstate = copy.deepcopy(self)
                newstate.move(color)
                (bc, newval) = newstate.a_star_search_rec(depth - 1, alpha, beta)
                if newval < value:
                    value = newval
                    bestcolor = color
                beta = max(beta, value)
                if beta <= alpha:
                    break
            return bestcolor, value

    def heuristic(self):
        return self.player1State.score - self.player2State.score
