import chess
import chess.engine

# Move evaluation function to assign a score to a given move
def move_eval(board, move):
    # Example move evaluation function (assigning a random score)
    import random
    return random.randint(-10, 10)

# Minimax algorithm with alpha-beta pruning
def minimax_alpha_beta(board: chess.Board, depth, alpha, beta, maximizing_player):
    if depth == 0:
        return 0

    if board.is_game_over():
        if board.result() == '1/2-1/2':
            return 0
        elif maximizing_player:
            return float('-inf')
        else:
            return float('inf')

    if maximizing_player:
        max_eval = float('-inf')
        for move in board.legal_moves:
            # board.push(move)
            move_score = move_eval(board, move)
            board.push(move)
            eval_score = move_eval(board, move) + minimax_alpha_beta(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval_score = -move_eval(board, move) + minimax_alpha_beta(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval

# Find the best move using minimax with alpha-beta pruning
def find_best_move(board, depth):
    best_move = None
    max_eval = float('-inf')
    alpha = float('-inf')
    beta = float('inf')
    for move in board.legal_moves:
        board.push(move)
        eval_score = move_eval(board, move) + minimax_alpha_beta(board, depth - 1, alpha, beta, False)
        board.pop()
        if eval_score > max_eval:
            max_eval = eval_score
            best_move = move
    return best_move
