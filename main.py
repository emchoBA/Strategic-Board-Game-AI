import copy
# Constants
AI_PIECE = '△'
HUMAN_PIECE = '○'
EMPTY = '.'
BOARD_SIZE = 7
MAX_MOVES = 50

TOTAL_AI_PIECES = 4
TOTAL_HUMAN_PIECES = 4


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


def check_death(board, row, col):
    """
    Checks for captured pieces in the specified row and column, based on:
    1. Wall Capture: Pieces next to the board edge are treated as walls.
    2. Normal Capture: Pieces sandwiched between two opponent pieces.
    """
    def capture_in_line(line, is_row=True):
        """
        Captures pieces in a single row or column based on Wall and Normal rules.
        """
        n = len(line)
        captured_indices = set()

        # Wall Capture: Start of the line
        if line[0] in [AI_PIECE, HUMAN_PIECE]:
            piece_type = line[0]
            opponent_type = AI_PIECE if piece_type == HUMAN_PIECE else HUMAN_PIECE
            i = 1
            while i < n and line[i] == piece_type:
                i += 1
            if i < n and line[i] == opponent_type:
                captured_indices.update(range(0, i))

        # Wall Capture: End of the line
        if line[-1] in [AI_PIECE, HUMAN_PIECE]:
            piece_type = line[-1]
            opponent_type = AI_PIECE if piece_type == HUMAN_PIECE else HUMAN_PIECE
            i = n - 2
            while i >= 0 and line[i] == piece_type:
                i -= 1
            if i >= 0 and line[i] == opponent_type:
                captured_indices.update(range(i + 1, n))

        # Normal Capture: Sandwiched between opponents
        i = 0
        while i < n:
            if line[i] not in [AI_PIECE, HUMAN_PIECE]:
                i += 1
                continue

            piece_type = line[i]
            opponent_type = AI_PIECE if piece_type == HUMAN_PIECE else HUMAN_PIECE

            start = i
            while i < n and line[i] == piece_type:
                i += 1

            if start > 0 and i < n and line[start - 1] == opponent_type and line[i] == opponent_type:
                captured_indices.update(range(start, i))

        # Remove captured pieces
        for idx in captured_indices:
            line[idx] = EMPTY

        return line

    # Check the specified row
    board[row] = capture_in_line(board[row])

    # Check the specified column
    column = [board[r][col] for r in range(BOARD_SIZE)]
    column = capture_in_line(column, is_row=False)
    for r in range(BOARD_SIZE):
        board[r][col] = column[r]

    return board



def move_piece(board, start_row, start_col, end_row, end_col):

    # Validate positions
    if not (0 <= start_row < BOARD_SIZE and 0 <= start_col < BOARD_SIZE):
        return False
    if not (0 <= end_row < BOARD_SIZE and 0 <= end_col < BOARD_SIZE):
        return False

    # Ensure there is a piece at the start and the end is empty
    if board[start_row][start_col] in [AI_PIECE, HUMAN_PIECE] and board[end_row][end_col] == EMPTY:
        # Ensure the move is horizontal or vertical (not diagonal)
        if (abs(start_row - end_row) == 1 and start_col == end_col) or \
           (abs(start_col - end_col) == 1 and start_row == end_row):
            # Perform the move
            piece = board[start_row][start_col]
            board[start_row][start_col] = EMPTY
            board[end_row][end_col] = piece
            return (end_row, end_col)  # Return the new position
    return False



def evaluate_board(board):
    ai_count = sum(row.count(AI_PIECE) for row in board)
    human_count = sum(row.count(HUMAN_PIECE) for row in board)
    return ai_count - human_count

def get_valid_moves(board, player_piece):

    # move is a tuple: ((start_row, start_col), (end_row, end_col)).
    moves = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == player_piece:
                # Check horizontal and vertical moves
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                for dr, dc in directions:
                    end_row, end_col = row + dr, col + dc
                    if 0 <= end_row < BOARD_SIZE and 0 <= end_col < BOARD_SIZE:
                        if board[end_row][end_col] == EMPTY:
                            moves.append(((row, col), (end_row, end_col)))
    return moves

def minimax(board, depth, is_maximizing, alpha, beta):

    if depth == 0 or all(row.count(AI_PIECE) == 0 for row in board) or all(row.count(HUMAN_PIECE) == 0 for row in board):
        return evaluate_board(board), None

    best_move = None

    if is_maximizing:
        max_eval = float('-inf')
        for move in get_valid_moves(board, AI_PIECE):
            new_board = copy.deepcopy(board)
            move_piece(new_board, *move[0], *move[1])
            new_board = check_death(new_board, *move[1])
            eval_score, _ = minimax(new_board, depth - 1, False, alpha, beta)
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in get_valid_moves(board, HUMAN_PIECE):
            new_board = copy.deepcopy(board)
            move_piece(new_board, *move[0], *move[1])
            new_board = check_death(new_board, *move[1])
            eval_score, _ = minimax(new_board, depth - 1, True, alpha, beta)
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_move

def ai_move(board, depth=8):
    temp, best_move = minimax(board, depth, True, float('-inf'), float('inf'))
    if best_move:
        move_piece(board, *best_move[0], *best_move[1])
        board = check_death(board, *best_move[1])
    return board

def main():
    board = initialize_board()
    display_board(board)

    print("AI is making the first move...")
    board = ai_move(board)
    display_board(board)

    while True:
        start_row, start_col = map(int, input("Enter start row and column: ").split())
        end_row, end_col = map(int, input("Enter end row and column: ").split())
        if move_piece(board, start_row, start_col, end_row, end_col):
            board = check_death(board, end_row, end_col)
            display_board(board)
        else:
            print("Invalid move! Try again.")
            continue

        print("AI is thinking...")
        board = ai_move(board)
        display_board(board)

        if all(row.count(AI_PIECE) == 0 for row in board):
            print("Human wins!")
            break
        elif all(row.count(HUMAN_PIECE) == 0 for row in board):
            print("AI wins!")
            break


if __name__ == "__main__":
    main()
