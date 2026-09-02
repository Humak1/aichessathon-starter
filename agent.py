import random
import time
import chess

# Import time runs once per game, inside a 60 second budget, before your clock starts.
# Load weights and build tables out here, not inside get_move.

class TimeoutError(Exception):
    pass

def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.

    fen           the position to move in; your colour is the side to move
    time_left_ms  your clock before this move, in milliseconds
    returns       "e2e4", or "e7e8q" for a promotion

    The process stays alive between your moves, so state you keep on a module or in a
    closure survives to the next call. It does not survive to the next game.

    print() is safe. Your stdout is redirected away from the protocol stream, discarded
    during rated games and shown back to you in the validation log.
    """
    board = chess.Board(fen)
    
    time_budget_ms = time_left_ms * 0.05 

    legal_move_list = list(board.legal_moves)
    if len(legal_move_list) == 1:
        time_budget_ms = 100
    
    time_budget_s = time_budget_ms / 1000.0
    start_time = time.time()
    best_move = None
    panic_best = random.choice(legal_move_list)
    
    for depth in range(1, 100): 
        best_score = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        deadline = start_time + time_budget_s
        
        depth_best_move = None
        depth_best_score = -float('inf')
        
        for move in board.legal_moves:
            if time.time() > deadline:
                raise TimeoutError

            board.push(move)
            try:
                score = -negmax(board, depth - 1, deadline, -beta, -alpha)
            except TimeoutError:
                board.pop()
                return best_move.uci() if best_move else panic_best.uci()
            
            board.pop()

            if score > depth_best_score:
                depth_best_score = score
                depth_best_move = move
                alpha = max(alpha, depth_best_score)

        best_move = depth_best_move

    return best_move.uci() if best_move else panic_best.uci()

def negmax(board, depth, deadline, alpha, beta):
    if time.time() > deadline:
        raise TimeoutError
    
    if depth == 0 or board.is_game_over():
        return evaluate(board)

    max_score = -float("inf")
    for move in board.legal_moves:
        board.push(move)
        score = -negmax(board, depth - 1, deadline, -beta, -alpha)
        board.pop()
        if score > max_score:
            max_score = score
            alpha = max(alpha, max_score)
        if alpha >= beta:
            break
    return max_score

def evaluate(board):
    if board.is_checkmate():
        return -99999

    if board.is_game_over():
        return 0
    
    score = 0
    value_dict = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }

    for piece in value_dict.keys():
        score += len(board.pieces(piece, chess.WHITE)) * value_dict[piece]
        score -= len(board.pieces(piece, chess.BLACK)) * value_dict[piece]

    return score if board.turn == chess.WHITE else -score
        

    # Everything from here down is yours to replace. baselines/greedy searches one ply,
    # baselines/minimax searches two. Neither is strong. Reading them is the fastest way
    # to see the shape of a search, and beating them is the first real milestone.
    # return random.choice(list(board.legal_moves)).uci()