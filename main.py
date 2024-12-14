import copy
# Constants
AI_PIECE = '△'
HUMAN_PIECE = '○'
EMPTY = '.'
BOARD_SIZE = 7
MAX_MOVES = 50

TOTAL_AI_PIECES = 4
TOTAL_HUMAN_PIECES = 4

GLOBAL_DEPTH = 4


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


def is_in_danger(board, row, col):
    # Check if a piece is in immediate danger of being captured
    piece = board[row][col]
    if piece == EMPTY:
        return False

    opponent = HUMAN_PIECE if piece == AI_PIECE else AI_PIECE

    # Check horizontal danger
    if col > 0 and col < BOARD_SIZE - 1:
        if board[row][col - 1] == opponent and board[row][col + 1] == opponent:
            return True
    # Edge capture check horizontal
    if col == 0 and board[row][1] == opponent:
        return True
    if col == BOARD_SIZE - 1 and board[row][BOARD_SIZE - 2] == opponent:
        return True

    # Check vertical danger
    if row > 0 and row < BOARD_SIZE - 1:
        if board[row - 1][col] == opponent and board[row + 1][col] == opponent:
            return True
    # Edge capture check vertical
    if row == 0 and board[1][col] == opponent:
        return True
    if row == BOARD_SIZE - 1 and board[BOARD_SIZE - 2][col] == opponent:
        return True

    return False


def can_capture(board, row, col):
    # Check if a piece can participate in capturing an opponent's piece
    piece = board[row][col]
    if piece == EMPTY:
        return False

    opponent = HUMAN_PIECE if piece == AI_PIECE else AI_PIECE

    # Check horizontal capture opportunities
    for c in range(BOARD_SIZE - 2):
        if board[row][c:c + 3].count(opponent) == 2 and board[row][c:c + 3].count(piece) == 1:
            return True

    # Check vertical capture opportunities
    column = [board[r][col] for r in range(BOARD_SIZE)]
    for r in range(BOARD_SIZE - 2):
        if column[r:r + 3].count(opponent) == 2 and column[r:r + 3].count(piece) == 1:
            return True

    return False


def get_territory_control(board):
    # Calculate territory control by counting accessible squares
    ai_territory = 0
    human_territory = 0

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == EMPTY:
                # Count nearby pieces
                ai_nearby = 0
                human_nearby = 0
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
                        if board[new_row][new_col] == AI_PIECE:
                            ai_nearby += 1
                        elif board[new_row][new_col] == HUMAN_PIECE:
                            human_nearby += 1

                if ai_nearby > human_nearby:
                    ai_territory += 1
                elif human_nearby > ai_nearby:
                    human_territory += 1

    return ai_territory - human_territory


def evaluate_board(board):
    """
    Enhanced evaluation function that considers:
    1. Piece count
    2. Pieces in danger
    3. Capturing opportunities
    4. Territory control
    5. Piece mobility
    """
    # Base piece count (most important)
    ai_count = sum(row.count(AI_PIECE) for row in board)
    human_count = sum(row.count(HUMAN_PIECE) for row in board)
    piece_score = (ai_count - human_count) * 100

    if ai_count == 0:
        return -10000  # Losing position
    if human_count == 0:
        return 10000  # Winning position

    # Count pieces in danger
    ai_danger = 0
    human_danger = 0
    # Count capturing opportunities
    ai_capture_ops = 0
    human_capture_ops = 0
    # Count piece mobility (available moves)
    ai_mobility = len(get_valid_moves(board, AI_PIECE))
    human_mobility = len(get_valid_moves(board, HUMAN_PIECE))

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == AI_PIECE:
                if is_in_danger(board, row, col):
                    ai_danger += 1
                if can_capture(board, row, col):
                    ai_capture_ops += 1
            elif board[row][col] == HUMAN_PIECE:
                if is_in_danger(board, row, col):
                    human_danger += 1
                if can_capture(board, row, col):
                    human_capture_ops += 1

    # Territory control
    territory_score = get_territory_control(board) * 10

    # Combine all factors with appropriate weights
    danger_score = (human_danger - ai_danger) * 50
    capture_score = (ai_capture_ops - human_capture_ops) * 30
    mobility_score = (ai_mobility - human_mobility) * 5

    # Final weighted score
    total_score = (
            piece_score +  # Base piece count (highest weight)
            danger_score +  # Pieces in danger
            capture_score +  # Capture opportunities
            territory_score +  # Territory control
            mobility_score  # Piece mobility
    )

    return total_score


def preliminary_evaluate_move(board, move):
    """Quick evaluation of a move for move ordering"""
    start_row, start_col = move[0]
    end_row, end_col = move[1]

    score = 0
    piece = board[start_row][start_col]

    # Prioritize moves that escape danger
    if is_in_danger(board, start_row, start_col):
        score += 50

    # Prioritize moves that can lead to captures
    temp_board = copy.deepcopy(board)
    move_piece(temp_board, start_row, start_col, end_row, end_col)
    if can_capture(temp_board, end_row, end_col):
        score += 30

    # Consider center control
    center_distance = abs(end_row - BOARD_SIZE // 2) + abs(end_col - BOARD_SIZE // 2)
    score -= center_distance  # Prefer moves closer to center

    return score


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

def ai_move(board, depth=GLOBAL_DEPTH, restricted_positions=None):

    if restricted_positions is None:
        restricted_positions = set()

    # Get all valid moves excluding restricted positions
    valid_moves = [move for move in get_valid_moves(board, AI_PIECE) if move[0] not in restricted_positions]

    if not valid_moves:
        return None

    # Find the best move using minimax
    best_eval = float('-inf')
    best_move = None
    for move in valid_moves:
        temp_board = copy.deepcopy(board)
        move_piece(temp_board, *move[0], *move[1])
        temp_board = check_death(temp_board, *move[1])
        eval_score, _ = minimax(temp_board, depth - 1, False, float('-inf'), float('inf'))
        if eval_score > best_eval:
            best_eval = eval_score
            best_move = move

    if best_move:
        move_piece(board, *best_move[0], *best_move[1])
        board = check_death(board, *best_move[1])
        print(f"AI moved from {best_move[0]} to {best_move[1]}")
        display_board(board)

    return best_move

def main():
    board = initialize_board()
    display_board(board)

    # AI starts first with two moves
    print("AI is making its first moves...")
    restricted_positions = set()
    for move in range(2):  # AI makes two moves
        first_move = ai_move(board, restricted_positions=restricted_positions)
        if first_move:
            restricted_positions.add(first_move[1])  # Restrict the end position of the first move

    while True:
        # Human's turn
        print("Your move! Enter start and end positions.")

        # Check the number of human pieces on the board
        human_pieces = sum(row.count(HUMAN_PIECE) for row in board)
        moves_to_make = 2 if human_pieces > 1 else 1

        # Add restrictions on user moves so that they cant move the same piece twice

        for move in range(moves_to_make):
            print(f"Move {move + 1} of {moves_to_make}")
            start_row, start_col = map(int, input("Enter start row and column: ").split())
            end_row, end_col = map(int, input("Enter end row and column: ").split())

            new_position = move_piece(board, start_row, start_col, end_row, end_col)
            if new_position:
                board = check_death(board, *new_position)
                display_board(board)
            else:
                print("Invalid move! Try again.")
                break  # End the turn if an invalid move is made

        # AI's turn
        print("AI is thinking...")
        restricted_positions = set()
        for move in range(2 if sum(row.count(AI_PIECE) for row in board) > 1 else 1):
            first_move = ai_move(board, restricted_positions=restricted_positions)
            if first_move:
                restricted_positions.add(first_move[1])  # Restrict the end position of the first move

        # Check for game end conditions
        if all(row.count(AI_PIECE) == 0 for row in board):
            print("Human wins!")
            break
        elif all(row.count(HUMAN_PIECE) == 0 for row in board):
            print("AI wins!")
            break


if __name__ == "__main__":
    main()