# Constants
AI_PIECE = '△'
HUMAN_PIECE = '○'
EMPTY = '.'
BOARD_SIZE = 7
MAX_MOVES = 50


# Initialize the 7x7 board with the starting configuration
def initialize_board():
    board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    # Set initial positions based on game rules
    # Place AI (triangle) pieces
    board[0][0] = AI_PIECE
    board[2][0] = AI_PIECE
    board[4][6] = AI_PIECE
    board[6][6] = AI_PIECE

    # Place Human (circle) pieces
    board[0][6] = HUMAN_PIECE
    board[2][6] = HUMAN_PIECE
    board[4][0] = HUMAN_PIECE
    board[6][0] = HUMAN_PIECE
    return board


# Display the board in CLI
def display_board(board):
    for row in board:
        print(' '.join(row))
    print("\n")


def check_piece(board, row, col):
    if board[row][col] == AI_PIECE:
        return 'AI'
    elif board[row][col] == HUMAN_PIECE:
        return 'Human'
    else:
        return 'Empty'


def move_piece(board, start_row, start_col, direction, player):
    if board[start_row][start_col] == EMPTY:
        print("No piece at the starting position.")
        return False
    # Add it so that player and AI can only move their own pieces
    # Maybe add new parameter like human or AI
    if player == 'AI' and board[start_row][start_col] == HUMAN_PIECE:
        print("AI can only move AI pieces.")
        return False
    elif player == 'Human' and board[start_row][start_col] == AI_PIECE:
        print("You can only move your own pieces.")
        return False

    if direction == 'up' and start_row > 0:
        new_row, new_col = start_row - 1, start_col
    elif direction == 'down' and start_row < BOARD_SIZE - 1:
        new_row, new_col = start_row + 1, start_col
    elif direction == 'left' and start_col > 0:
        new_row, new_col = start_row, start_col - 1
    elif direction == 'right' and start_col < BOARD_SIZE - 1:
        new_row, new_col = start_row, start_col + 1
    else:
        print("Invalid move.")
        return False

    if check_piece(board, new_row, new_col) == 'Empty':
        board[new_row][new_col] = board[start_row][start_col]
        board[start_row][start_col] = EMPTY
        print("Move successful, " + player + " moved piece from (" + str(start_row) + ", " + str(start_col) + ") to (" +
              str(new_row) + ", " + str(new_col) + ").")
        return True
    else:
        print("Target position is not empty.")
        return False


def main():
    board = initialize_board()
    display_board(board)
    print("\n")
    move_piece(board, 0, 0, 'down', 'AI')
    display_board(board)
    move_piece(board, 0, 0, 'right', 'AI')
    display_board(board)
    move_piece(board, 0, 6, 'down', 'Human')
    display_board(board)
    move_piece(board, 1, 0, 'right', 'Human')
    display_board(board)


# Start the game
if __name__ == "__main__":
    main()
