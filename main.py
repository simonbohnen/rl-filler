from pygame.locals import QUIT, KEYDOWN, K_RETURN, MOUSEBUTTONDOWN

from helpers import *
from fillerstate import FillerState


def main(policy=None):
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
    board = [[-1 for _ in range(HEIGHT)] for _ in range(WIDTH)] if initing else get_random_regular_board()
    state = None if initing else FillerState(board, 1, False)

    draw_board(board, background)
    print(str_board(board))
    print_colors()

    while running:
        clock.tick(60)

        # Handle Input Events
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if initing and event.key == K_RETURN:
                    # Select next color when drawing the board
                    color += 1
                    if color == len(COLORS):
                        initing = False
                        state = FillerState(board, 1, False)
                if event.key in COLOR_KEYS:
                    # perform move using number keys
                    color = COLOR_KEYS.index(event.key) % 6
                    state.move(color)
                    if not state.is_final_state:
                        state.do_alpha_move()
                    if state.is_final_state:
                        print("END: Finished game. Player 1 scored %d, player 2 scored %d."
                              % (state.player1State.score, state.player2State.score))
                        board = get_random_regular_board()
                        state = FillerState(board, 1, False)
            elif event.type == MOUSEBUTTONDOWN:
                if initing:
                    # set color of board
                    set_color(board, pygame.mouse.get_pos(), color)
                else:
                    # move with color that was clicked on
                    state.move(get_color(state.board, pygame.mouse.get_pos()))
                    if not state.is_final_state:
                        if policy:
                            state.move(policy.action())
                        state.do_alpha_move()
                    if state.is_final_state:
                        print("END: Finished game. Player 1 scored %d, player 2 scored %d."
                              % (state.player1State.score, state.player2State.score))
                        board = get_random_regular_board()
                        state = FillerState(board, 1, False)

        draw_board(board, background)
        screen.blit(background, (0, 0))
        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    main()
