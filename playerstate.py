from defs import *
from typing import List


class PlayerState:
    def __init__(self, board: Board, isplayer1: int):
        """
        Creates a new PlayerState. Should only be called by FillerState.__init__.
        :param board: the game's initial board.
        :param isplayer1: indicates whether this is the PlayerState for player 1.
        """
        self.adjacents: List[Cellset] = [set() for _ in range(6)]
        self.score: int = 1
        self.owned: Cellset = set()
        if isplayer1:
            self.owned.add((0, HEIGHT - 1))
            self.adjacents[board[0][HEIGHT - 2]].add((0, HEIGHT - 2))
            self.adjacents[board[1][HEIGHT - 1]].add((1, HEIGHT - 1))
        else:
            self.owned.add((WIDTH - 1, 0))
            self.adjacents[board[WIDTH - 1][1]].add((WIDTH - 1, 1))
            self.adjacents[board[WIDTH - 2][0]].add((WIDTH - 2, 0))

    def move(self, board: Board, color: int, is_available: Field,
             owned_by_other: Cellset, quiet: bool = True) -> Cellset:
        """
        Performs the move indicated by `color` on the board. This is includes filling enclaves that form.
        Should only be called by FillerState.move.
        :param board: The board to perform the move on.
        :param color: The color chosen for the move.
        :param is_available: A field indicating which cells are not owned by either player yet.
                            Gets updated to reflect move.
        :param owned_by_other: A cellset containing the cells owned by the other player. Used for finding enclaves.
        :param quiet: Whether printing should be quiet.
        :return: set of newly owned cells that are not enclaves.
        """
        for x, y in self.owned:
            board[x][y] = color

        self.owned |= self.adjacents[color]
        self.score += len(self.adjacents[color])
        if not quiet:
            print("Added %d adjacents" % len(self.adjacents[color]))
        new_owned: Cellset = self.adjacents[color]
        for x, y in new_owned:
            is_available[x][y] = False
        self.adjacents[color] = set()
        self.__update_adjacents(board, new_owned, is_available)

        # Update enclaves
        enclaves = self.__get_enclaves(owned_by_other)
        for x, y in enclaves:
            board[x][y] = color
            is_available[x][y] = False
        self.score += len(enclaves)
        if not quiet:
            print("Added %d enclaves" % len(enclaves))
        self.owned |= enclaves
        for c in range(len(COLORS)):
            self.adjacents[c] -= enclaves

        return new_owned

    def __update_adjacents(self, board: Board, new_owned: Cellset, is_available: Field):
        """
        Updates the sets containing adjacent cells of each color. Only called by move.
        :param board: the board.
        :param new_owned: the newly owned cells after the last move
        :param is_available: a field indicating which cells are still unoccupied.
        """
        to_add = []
        for x, y in new_owned:
            to_add += [(x+1, y), (x, y+1), (x-1, y), (x, y-1)]

        to_add_set: Cellset = set([(x, y) for x, y in to_add
                                   if 0 <= x < WIDTH and 0 <= y < HEIGHT and is_available[x][y]])
        for x, y in to_add_set:
            self.adjacents[board[x][y]].add((x, y))

    def remove_new_owned_from_adjacents(self, new_owned: Cellset):
        """
        Removes the newly owned cells of the other player from the adjacency sets of this player.
        Only called by FillerState.move
        :param new_owned: The newly owned cells of the other player.
        """
        for c in range(len(COLORS)):
            self.adjacents[c] -= new_owned

    def __explore_cell(self, x: int, y: int, owned_by_other: Cellset,
                       open_cells: List[Point], checked: Field, possible_enclaves: Cellset, enclaves) -> int:
        """
        Explores a cell for finding enclaves. Is called for neighbors recursively.
        Only used by __get_enclaves.
        :param x: the x coordinate of the cell to explore
        :param y: the y coordinate of the cell to explore
        :param owned_by_other: cells owned by the other player
        :param open_cells: unoccupied cells
        :param checked: cells that have been checked by this method. gets updated.
        :param possible_enclaves: cells that could be enclaves. gets updated
        :param enclaves: cells that have been confirmed as enclaves
        :return: whether this cell is an enclave as indicated by the constants defined at the beginning.
        """
        # noinspection PyShadowingBuiltins
        open = 0
        possible = 1
        owned = 2
        enclave = 3

        pos = (x, y)

        if not 0 <= x < WIDTH or not 0 <= y < HEIGHT or pos in self.owned:
            # Out of bounds or own cell.
            return owned
        if pos in enclaves:
            # Is already enclave, don't need to explore.
            return enclave
        if pos in owned_by_other or pos in open_cells:
            # Found way out.
            possible_enclaves.clear()
            return open
        if checked[x][y]:
            return possible
        checked[x][y] = True
        # Explore.
        neighbors = [(x+1, y), (x, y-1), (x-1, y), (x, y+1)]
        for nx, ny in neighbors:
            result = self.__explore_cell(nx, ny, owned_by_other, open_cells, checked, possible_enclaves, enclaves)
            if result == owned or result == possible:
                # Neighbor is out of bounds or has been checked already.
                continue
            else:
                # Neighbor has found open cell.
                open_cells.append(pos)
                return open
        # Only out of bounds or own cells or checked were found.
        possible_enclaves.add(pos)
        return possible

    def __get_enclaves(self, owned_by_other):
        """
        Gets the latest enclaves after a move. called by move.
        :param owned_by_other: the cells owned by the other player
        :return: The newly found enclaves to be filled.
        """
        open_cells = []
        enclaves: Cellset = set()
        for c in range(len(COLORS)):
            for x, y in self.adjacents[c]:
                possible_enclaves: Cellset = set()
                self.__explore_cell(x, y, owned_by_other, open_cells,
                                    [[False for _ in range(HEIGHT)] for _ in range(WIDTH)], possible_enclaves, enclaves)
                enclaves |= possible_enclaves
        return enclaves
