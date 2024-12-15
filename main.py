import sys
sys.setrecursionlimit(10 ** 7)

# ---------------------------
#           CONSTANTS
# ---------------------------
AI_PIECE = '△'
HUMAN_PIECE = '○'
EMPTY = '.'
BOARD_SIZE = 7

GLOBAL_DEPTH = 4
MOVES = 50

# A global cache for board states in minimax (transposition table)
transposition_table = {}


# ---------------------------
#        INITIALIZATION
# ---------------------------
def initialize_board():
    """
    Create and return a 7x7 board with the initial distribution of AI and Human pieces.
    """
    board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

    # Place AI pieces
    board[0][0] = AI_PIECE
    board[2][0] = AI_PIECE
    board[4][6] = AI_PIECE
    board[6][6] = AI_PIECE

    # Place Human pieces
    board[0][6] = HUMAN_PIECE
    board[2][6] = HUMAN_PIECE
    board[4][0] = HUMAN_PIECE
    board[6][0] = HUMAN_PIECE

    return board


def display_board(board):
    """
    Print the 7x7 board with spacing.
    """
    for row in board:
        print(' '.join(row))
    print()


# ---------------------------
#         CORE LOGIC
# ---------------------------
def check_death(board, row, col):
    """
    Check for captures in the row and column of the piece that just moved to (row, col).
    Returns a 'revert' function that can restore the board to its original state.

    Capture rules:
      - Wall capture: If a contiguous block of pieces is adjacent to the board edge
        and is then immediately followed by an opponent piece, those pieces are captured.
      - Normal capture: A contiguous block of pieces is 'sandwiched' between opponent pieces.
    """

    def capture_in_line(line):
        """
        Given a list `line` (representing a row or column), remove captured pieces in-place.
        Returns the updated line.
        """
        n = len(line)
        captured_indices = set()

        # -- Wall Capture at the START of the line --
        if line[0] in (AI_PIECE, HUMAN_PIECE):
            current_piece = line[0]
            opponent = AI_PIECE if current_piece == HUMAN_PIECE else HUMAN_PIECE
            i = 1
            while i < n and line[i] == current_piece:
                i += 1
            if i < n and line[i] == opponent:
                # Everything from index 0 up to (i-1) is captured
                captured_indices.update(range(0, i))

        # -- Wall Capture at the END of the line --
        if line[-1] in (AI_PIECE, HUMAN_PIECE):
            current_piece = line[-1]
            opponent = AI_PIECE if current_piece == HUMAN_PIECE else HUMAN_PIECE
            i = n - 2
            while i >= 0 and line[i] == current_piece:
                i -= 1
            if i >= 0 and line[i] == opponent:
                # Everything from (i+1) up to the end is captured
                captured_indices.update(range(i + 1, n))

        # -- Normal "Sandwich" Capture --
        i = 0
        while i < n:
            if line[i] not in (AI_PIECE, HUMAN_PIECE):
                i += 1
                continue

            current_piece = line[i]
            opponent = AI_PIECE if current_piece == HUMAN_PIECE else HUMAN_PIECE

            # Find the contiguous block of the same pieces
            start_block = i
            while i < n and line[i] == current_piece:
                i += 1

            # Now line[start_block..(i-1)] is the contiguous block
            if start_block > 0 and i < n:
                if line[start_block - 1] == opponent and line[i] == opponent:
                    captured_indices.update(range(start_block, i))

        # Remove captured pieces
        for idx in captured_indices:
            line[idx] = EMPTY

        return line

    # --- SNAPSHOT row and column (for undo later) ---
    original_row_state = board[row][:]
    original_col_state = [board[r][col] for r in range(BOARD_SIZE)]

    # --- Check capture in row ---
    board[row] = capture_in_line(board[row])

    # --- Check capture in column ---
    column = [board[r][col] for r in range(BOARD_SIZE)]
    column = capture_in_line(column)
    for r in range(BOARD_SIZE):
        board[r][col] = column[r]

    # Return a function that can revert the board back to its original state
    def revert():
        board[row] = original_row_state
        for r in range(BOARD_SIZE):
            board[r][col] = original_col_state[r]

    return revert


def move_piece(board, start_row, start_col, end_row, end_col):
    """
    Move a piece from (start_row, start_col) to (end_row, end_col) if valid.
    Returns the ending position (end_row, end_col) if move is successful, or None if invalid.
    """
    # Validate positions
    if not (0 <= start_row < BOARD_SIZE and 0 <= start_col < BOARD_SIZE):
        return None
    if not (0 <= end_row < BOARD_SIZE and 0 <= end_col < BOARD_SIZE):
        return None

    piece = board[start_row][start_col]
    if piece not in (AI_PIECE, HUMAN_PIECE):
        return None  # no piece to move

    if board[end_row][end_col] != EMPTY:
        return None  # target must be empty

    # Only horizontal or vertical single-step moves allowed
    if (abs(start_row - end_row) == 1 and start_col == end_col) or \
            (abs(start_col - end_col) == 1 and start_row == end_row):
        board[start_row][start_col] = EMPTY
        board[end_row][end_col] = piece
        return (end_row, end_col)

    return None


def is_in_danger(board, row, col):
    """
    Check if the piece at (row,col) is immediately in danger of capture
    by either wall-capture or normal sandwich-capture in the next move.
    """
    piece = board[row][col]
    if piece == EMPTY:
        return False

    opponent = HUMAN_PIECE if piece == AI_PIECE else AI_PIECE

    # Horizontal Danger Check
    if 0 < col < BOARD_SIZE - 1:
        if board[row][col - 1] == opponent and board[row][col + 1] == opponent:
            return True
    if col == 0 and board[row][1] == opponent:
        return True
    if col == BOARD_SIZE - 1 and board[row][BOARD_SIZE - 2] == opponent:
        return True

    # Vertical Danger Check
    if 0 < row < BOARD_SIZE - 1:
        if board[row - 1][col] == opponent and board[row + 1][col] == opponent:
            return True
    if row == 0 and board[1][col] == opponent:
        return True
    if row == BOARD_SIZE - 1 and board[BOARD_SIZE - 2][col] == opponent:
        return True

    return False


def can_capture(board, row, col):
    """
    Check if the piece at (row,col) can help capture an opponent
    on its next move (by forming a sandwich or using wall-capture).
    """
    piece = board[row][col]
    if piece == EMPTY:
        return False

    opponent = HUMAN_PIECE if piece == AI_PIECE else AI_PIECE

    # Check horizontally in the row
    for c in range(BOARD_SIZE - 2):
        window = board[row][c:c + 3]
        if window.count(opponent) == 2 and window.count(piece) == 1:
            return True

    # Check vertically in the column
    column = [board[r][col] for r in range(BOARD_SIZE)]
    for r in range(BOARD_SIZE - 2):
        window = column[r:r + 3]
        if window.count(opponent) == 2 and window.count(piece) == 1:
            return True

    return False


def get_territory_control(board):
    """
    Returns the difference in territory control: (# squares closer to AI) - (# squares closer to Human).

    A square is "controlled" by AI if it has more AI neighbors
    (up, down, left, right) than Human neighbors, and vice versa.
    """
    ai_territory = 0
    human_territory = 0

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == EMPTY:
                ai_neighbors = 0
                human_neighbors = 0
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                        neighbor = board[nr][nc]
                        if neighbor == AI_PIECE:
                            ai_neighbors += 1
                        elif neighbor == HUMAN_PIECE:
                            human_neighbors += 1

                if ai_neighbors > human_neighbors:
                    ai_territory += 1
                elif human_neighbors > ai_neighbors:
                    human_territory += 1

    return ai_territory - human_territory


def get_valid_moves(board, player_piece):
    """
    Return a list of valid single-step moves for the given player_piece.
    Each move is ((start_row, start_col), (end_row, end_col)).
    """
    moves = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == player_piece:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                        if board[nr][nc] == EMPTY:
                            moves.append(((row, col), (nr, nc)))
    return moves


def preliminary_evaluate_move(board, move):
    """
    A quick heuristic score for move ordering:
      - +50 if it escapes danger
      - +30 if it can immediately lead to a capture
      - -distance_from_center to prefer moving closer to center
    """
    (sr, sc), (er, ec) = move
    # piece = board[sr][sc]

    score = 0
    # If the piece is currently in danger, moving it away is beneficial
    if is_in_danger(board, sr, sc):
        score += 50

    # Temporarily make the move in-place
    original_piece = board[sr][sc]
    original_destination = board[er][ec]

    board[sr][sc] = EMPTY
    board[er][ec] = original_piece

    # If after moving, this piece can capture, that is beneficial
    if can_capture(board, er, ec):
        score += 30

    # Revert the move
    board[er][ec] = original_destination
    board[sr][sc] = original_piece

    # Bonus for being closer to center
    center_distance = abs(er - BOARD_SIZE // 2) + abs(ec - BOARD_SIZE // 2)
    score -= center_distance

    return score


def evaluate_board(board):
    """
    Comprehensive evaluation that returns a numeric score
    favoring the AI (△) over the Human (○).

    Considerations:
      - Piece count difference
      - Pieces in danger
      - Capture opportunities
      - Territory control
      - Mobility (valid moves)
    """
    # Count pieces
    ai_piece_count = sum(row.count(AI_PIECE) for row in board)
    human_piece_count = sum(row.count(HUMAN_PIECE) for row in board)

    # Terminal states
    if ai_piece_count == 0:
        return -10000  # AI lost
    if human_piece_count == 0:
        return 10000  # AI won

    # Base piece difference
    piece_score = (ai_piece_count - human_piece_count) * 100

    # Precompute moves for both sides to avoid repeated calls
    ai_moves = get_valid_moves(board, AI_PIECE)
    human_moves = get_valid_moves(board, HUMAN_PIECE)
    ai_mobility = len(ai_moves)
    human_mobility = len(human_moves)

    # Danger & capture checks
    ai_danger = 0
    human_danger = 0
    ai_capture_ops = 0
    human_capture_ops = 0

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == AI_PIECE:
                if is_in_danger(board, r, c):
                    ai_danger += 1
                if can_capture(board, r, c):
                    ai_capture_ops += 1
            elif board[r][c] == HUMAN_PIECE:
                if is_in_danger(board, r, c):
                    human_danger += 1
                if can_capture(board, r, c):
                    human_capture_ops += 1

    territory_score = get_territory_control(board) * 10
    danger_score = (human_danger - ai_danger) * 50
    capture_score = (ai_capture_ops - human_capture_ops) * 30
    mobility_score = (ai_mobility - human_mobility) * 5

    total_score = piece_score + danger_score + capture_score + territory_score + mobility_score
    return total_score


def board_to_key(board):
    """
    Convert the 2D board into an immutable, hashable form (tuple of tuples).
    Used for transposition table lookups.
    """
    return tuple(tuple(row) for row in board)


def minimax(board, depth, is_maximizing, alpha, beta):
    """
    Alpha-Beta Minimax with in-place move, capture-undo, and a transposition table to cache results.
    Returns (best_score, best_move).
    best_move is ((start_row, start_col), (end_row, end_col)).
    """

    # Check for terminal or depth limit
    ai_exists = any(AI_PIECE in row for row in board)
    human_exists = any(HUMAN_PIECE in row for row in board)
    if depth == 0 or not ai_exists or not human_exists:
        return evaluate_board(board), None

    # Check transposition table
    state_key = board_to_key(board)
    if state_key in transposition_table:
        cached = transposition_table[state_key]
        # Use cached result if it meets depth requirement and alpha/beta window
        if cached['depth'] >= depth and cached['alpha'] <= alpha and cached['beta'] >= beta:
            return cached['score'], cached['move']

    best_move = None

    if is_maximizing:
        # Maximizing for AI
        all_moves = get_valid_moves(board, AI_PIECE)
        if not all_moves:
            return evaluate_board(board), None  # No moves -> evaluate board

        # Move ordering (descending by preliminary evaluation)
        all_moves.sort(key=lambda mv: preliminary_evaluate_move(board, mv), reverse=True)

        max_eval = float('-inf')

        for mv in all_moves:
            (sr, sc), (er, ec) = mv

            # In-place move
            piece = board[sr][sc]
            board[sr][sc] = EMPTY
            board[er][ec] = piece

            # In-place capture
            revert_capture = check_death(board, er, ec)

            # Recurse minimax
            eval_score, _ = minimax(board, depth - 1, False, alpha, beta)

            # Undo capture
            revert_capture()
            # Undo move
            board[er][ec] = EMPTY
            board[sr][sc] = piece

            # Track best move
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = mv

            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break

        # Store in transposition table
        transposition_table[state_key] = {
            'score': max_eval,
            'move': best_move,
            'depth': depth,
            'alpha': alpha,
            'beta': beta
        }
        return max_eval, best_move

    else:
        # Minimizing for HUMAN
        all_moves = get_valid_moves(board, HUMAN_PIECE)
        if not all_moves:
            return evaluate_board(board), None

        # Move ordering (ascending by preliminary evaluation)
        all_moves.sort(key=lambda mv: preliminary_evaluate_move(board, mv))

        min_eval = float('inf')

        for mv in all_moves:
            (sr, sc), (er, ec) = mv

            piece = board[sr][sc]
            board[sr][sc] = EMPTY
            board[er][ec] = piece

            revert_capture = check_death(board, er, ec)

            eval_score, _ = minimax(board, depth - 1, True, alpha, beta)

            revert_capture()
            board[er][ec] = EMPTY
            board[sr][sc] = piece

            if eval_score < min_eval:
                min_eval = eval_score
                best_move = mv

            beta = min(beta, eval_score)
            if beta <= alpha:
                break

        transposition_table[state_key] = {
            'score': min_eval,
            'move': best_move,
            'depth': depth,
            'alpha': alpha,
            'beta': beta
        }
        return min_eval, best_move


def ai_move(board, depth=GLOBAL_DEPTH, restricted_positions=None):
    """
    AI finds and executes its best move on the board using minimax.
    restricted_positions: set of positions (row,col) that AI cannot re-move in the same turn.
    Returns the best move chosen.
    """
    if restricted_positions is None:
        restricted_positions = set()

    possible_moves = [mv for mv in get_valid_moves(board, AI_PIECE) if mv[0] not in restricted_positions]
    if not possible_moves:
        return None

    # Move ordering for AI
    possible_moves.sort(key=lambda mv: preliminary_evaluate_move(board, mv), reverse=True)

    best_eval = float('-inf')
    best_move = None

    for mv in possible_moves:
        (sr, sc), (er, ec) = mv

        # In-place move
        piece = board[sr][sc]
        board[sr][sc] = EMPTY
        board[er][ec] = piece

        revert_capture = check_death(board, er, ec)

        eval_score, _ = minimax(board, depth - 1, False, float('-inf'), float('inf'))

        revert_capture()
        board[er][ec] = EMPTY
        board[sr][sc] = piece

        if eval_score > best_eval:
            best_eval = eval_score
            best_move = mv

    # Actually execute the best move found
    if best_move:
        (sr, sc), (er, ec) = best_move
        final_position = move_piece(board, sr, sc, er, ec)
        if final_position:
            check_death(board, er, ec)  # finalize captures
            print(f"AI moved from {best_move[0]} to {best_move[1]}")
            display_board(board)

    return best_move


# ---------------------------
#           MAIN LOOP
# ---------------------------
def main():
    game_board = initialize_board()
    display_board(game_board)

    moves_left = MOVES

    print("AI is making its first moves...")
    restricted = set()
    # AI gets two moves initially
    for _ in range(2):
        first_move = ai_move(game_board, GLOBAL_DEPTH, restricted_positions=restricted)
        if first_move:
            moves_left -= 1  # AI made a move so decrement moves left
            restricted.add(first_move[1])  # AI can't move the same piece again this turn

            if moves_left <= 0:
                break

    # Main game loop
    while True:

        if moves_left <= 0:
            break
        print("It's your turn! Please enter moves.")

        # Human might get 2 moves if more than 1 piece remains
        human_pieces = sum(row.count(HUMAN_PIECE) for row in game_board)
        moves_to_make = 2 if human_pieces > 1 else 1

        # Keep track of restricted piece positions so human can't move the same piece twice
        restricted = set()

        for move_index in range(moves_to_make):
            print(f"Human move {move_index + 1} of {moves_to_make}")

            try:
                start_r, start_c = map(int, input("Enter start (row,col): ").split())
                end_r, end_c = map(int, input("Enter end   (row,col): ").split())
            except ValueError:
                print("Invalid input. Try again.")
                # Let the user retry the same move
                move_index -= 1
                continue

            # Check if user is trying to move the same piece again
            if (start_r, start_c) in restricted:
                print("You can't move the same piece twice in a turn! Try again.")
                move_index -= 1
                continue

            # Attempt the move
            final_pos = move_piece(game_board, start_r, start_c, end_r, end_c)
            if final_pos:

                moves_left -= 1  # Human made a move so decrement moves left

                revert_fn = check_death(game_board, *final_pos)
                revert_fn()  # finalize captures
                display_board(game_board)
                restricted.add(final_pos)  # can't move this piece again this turn
            else:
                print("Invalid move! Try again.")
                move_index -= 1
                continue

            if moves_left <= 0:
                break

        # Double check out of loop (nolur nolmaz)
        if moves_left <= 0:
            break

        # Check win conditions
        if all(AI_PIECE not in row for row in game_board):
            print("Human wins!")
            return
        elif all(HUMAN_PIECE not in row for row in game_board):
            print("AI wins!")
            return

        # AI Turn
        print("AI is thinking...")
        ai_pieces = sum(row.count(AI_PIECE) for row in game_board)
        ai_moves_count = 2 if ai_pieces > 1 else 1

        restricted = set()
        for _ in range(ai_moves_count):
            move_result = ai_move(game_board, GLOBAL_DEPTH, restricted_positions=restricted)
            if move_result:
                moves_left -= 1  # AI made a move so decrement moves left
                restricted.add(move_result[1])  # AI can't move the same piece again this turn

                if moves_left <= 0:
                    break

        # Triple check out of loop (nolur nolmaz)
        if moves_left <= 0:
            break

        # Check win conditions again
        if all(AI_PIECE not in row for row in game_board):
            print("Human wins!")
            return
        elif all(HUMAN_PIECE not in row for row in game_board):
            print("AI wins!")
            return

    # Game over, no more moves left
    print("No more moves left!")
    ai_piece_count = sum(row.count(AI_PIECE) for row in game_board)
    human_piece_count = sum(row.count(HUMAN_PIECE) for row in game_board)

    if ai_piece_count > human_piece_count:
        print("AI wins by piece count!")
    elif human_piece_count > ai_piece_count:
        print("Human wins by piece count!")
    else:
        print("It's a draw!")


if __name__ == "__main__":
    main()
