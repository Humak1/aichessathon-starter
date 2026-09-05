"""Compact, strong classical chess engine for AI Chessathon.

The engine intentionally keeps the algorithm small and auditable:
iterative deepening, PVS/alpha-beta, a transposition table, TT/killer/history/
counter-move ordering, conservative null move, LMR, check extensions, and
tactical quiescence.

No external engine, network, subprocess, native extension, or precomputed
engine evaluations are used.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import chess

# ---------------------------------------------------------------------------
# Values / search constants
# ---------------------------------------------------------------------------

PAWN = 100
KNIGHT = 320
BISHOP = 330
ROOK = 500
QUEEN = 900
KING = 20_000

VALUES = {
    chess.PAWN: PAWN,
    chess.KNIGHT: KNIGHT,
    chess.BISHOP: BISHOP,
    chess.ROOK: ROOK,
    chess.QUEEN: QUEEN,
    chess.KING: KING,
}

MATE = 100_000
MATE_BOUND = 90_000
MAX_DEPTH = 64
MAX_PLY = 96
QMAX = 10

# Time: use substantially more than the old 5-second hard cap, but protect
# the clock aggressively. At 120s this targets ~8s; at low time it contracts.
MOVE_CAP_MS = 9_000
RESERVE_MS = 700
MIN_BUDGET_MS = 80

# Clock checks are cheap enough every 512 nodes in normal time, more often
# when the budget is small.
NORMAL_CHECK_MASK = 511
FAST_CHECK_MASK = 63

TT_MAX = 300_000

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

# MVV-LVA / ordering bands.
TT_ORDER = 2_000_000
GOOD_CAPTURE = 1_000_000
KILLER1 = 800_000
KILLER2 = 700_000
COUNTER = 600_000
PROMOTION_ORDER = 500_000

# ---------------------------------------------------------------------------
# Piece square tables
# ---------------------------------------------------------------------------

PAWN_PST = (
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
)

KNIGHT_PST = (
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
)

BISHOP_PST = (
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
)

ROOK_PST = (
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    0, 0, 0, 5, 5, 0, 0, 0,
)

QUEEN_PST = (
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
)

KING_MG = (
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20,
)

KING_EG = (
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10, 0, 0, -10, -20, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -30, 0, 0, 0, 0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
)

PSTS = {
    chess.PAWN: PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.QUEEN: QUEEN_PST,
}

PHASE_MAX = 2 * (2 * KNIGHT + 2 * BISHOP + 2 * ROOK + QUEEN)


class SearchTimeout(Exception):
    pass


@dataclass(slots=True)
class TTEntry:
    depth: int
    score: int
    flag: int
    move: chess.Move | None


@dataclass(slots=True)
class SearchState:
    deadline: float
    check_mask: int
    # Positions visited on the current search path (root to current node),
    # always provided by get_move at construction time. Kept required rather
    # than Optional so every call site can use it directly without a guard.
    path: set[Any]
    nodes: int = 0


# Persistent across moves in the same game. TT is deliberately retained:
# positions recur often, and this is exactly what a classical engine wants.
TT: dict[Any, TTEntry] = {}

# Search heuristics persist too; this is safe because they are ordering data,
# not position evaluations.
KILLERS: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY)]
HISTORY = [0] * 4096
COUNTERMOVE: list[chess.Move | None] = [None] * 4096

# Actual-game positions we have observed. Only used to avoid voluntarily
# entering known threefold repetitions. NEVER incorporated into evaluation.
GAME_REPETITIONS: dict[Any, int] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pos_key(board: chess.Board) -> Any:
    """Fast python-chess position key (includes side, castling, ep)."""
    return board._transposition_key()


def tt_key(board: chess.Board) -> tuple[Any, int, int]:
    # Draw state is part of the search state. The position key carries the
    # board/side/castling/ep state; halfmove is capped at 100 because 100 is
    # the only threshold that changes the immediate game result. The observed
    # repetition count is capped at 2 because 2 -> next occurrence is a draw.
    k = pos_key(board)
    halfmove = min(100, board.halfmove_clock)
    rep = min(2, GAME_REPETITIONS.get(k, 0))
    return k, halfmove, rep


def touch_clock(state: SearchState) -> None:
    state.nodes += 1
    if (state.nodes & state.check_mask) == 0 and time.perf_counter() >= state.deadline:
        raise SearchTimeout


def hist_index(move: chess.Move) -> int:
    return (move.from_square << 6) | move.to_square


def update_history(move: chess.Move, depth: int, good: bool = True) -> None:
    idx = hist_index(move)
    bonus = min(12_000, depth * depth * 32)
    if good:
        HISTORY[idx] = min(100_000, HISTORY[idx] + bonus)
    else:
        HISTORY[idx] = max(-100_000, HISTORY[idx] - max(8, bonus // 2))


def store_killer(ply: int, move: chess.Move) -> None:
    if ply >= MAX_PLY:
        return
    if KILLERS[ply][0] != move:
        KILLERS[ply][1] = KILLERS[ply][0]
        KILLERS[ply][0] = move


def capture_value(board: chess.Board, move: chess.Move) -> int:
    if board.is_en_passant(move):
        return PAWN
    victim = board.piece_type_at(move.to_square)
    return 0 if victim is None else VALUES[victim]


def is_quiet(board: chess.Board, move: chess.Move) -> bool:
    return not board.is_capture(move) and move.promotion is None


def move_score(
    board: chess.Board,
    move: chess.Move,
    tt_move: chess.Move | None,
    ply: int,
    prev_move: chess.Move | None,
) -> int:
    if tt_move is not None and move == tt_move:
        return TT_ORDER

    score = HISTORY[hist_index(move)]

    if ply < MAX_PLY:
        if move == KILLERS[ply][0]:
            score += KILLER1
        elif move == KILLERS[ply][1]:
            score += KILLER2

    if prev_move is not None:
        cm = COUNTERMOVE[hist_index(prev_move)]
        if cm is not None and move == cm:
            score += COUNTER

    if move.promotion is not None:
        score += PROMOTION_ORDER + VALUES.get(move.promotion, QUEEN)

    if board.is_capture(move):
        victim = capture_value(board, move)
        attacker_type = board.piece_type_at(move.from_square)
        # A legal move always has a piece on from_square, but mypy can't
        # prove that statically, so fall back to a pawn's value defensively.
        attacker = VALUES[attacker_type] if attacker_type is not None else PAWN
        # MVV-LVA, with an extra preference for checks.
        score += GOOD_CAPTURE + 16 * victim - attacker
        if board.gives_check(move):
            score += 12_000
        return score

    if board.gives_check(move):
        score += 100_000
    if board.is_castling(move):
        score += 2_000
    return score


def order_moves(
    board: chess.Board,
    tt_move: chess.Move | None,
    ply: int,
    prev_move: chess.Move | None,
) -> list[chess.Move]:
    moves = list(board.legal_moves)
    if len(moves) <= 1:
        return moves
    moves.sort(
        key=lambda m: move_score(board, m, tt_move, ply, prev_move),
        reverse=True,
    )
    return moves


def mate_to_tt(score: int, ply: int) -> int:
    if score >= MATE_BOUND:
        return score + ply
    if score <= -MATE_BOUND:
        return score - ply
    return score


def mate_from_tt(score: int, ply: int) -> int:
    if score >= MATE_BOUND:
        return score - ply
    if score <= -MATE_BOUND:
        return score + ply
    return score


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def pawn_structure(board: chess.Board, color: chess.Color) -> int:
    pawns = board.pieces(chess.PAWN, color)
    if not pawns:
        return 0

    files = [0] * 8
    for sq in pawns:
        files[chess.square_file(sq)] += 1

    score = 0
    for f, count in enumerate(files):
        if count > 1:
            score -= 12 * (count - 1)
        if count:
            left = files[f - 1] if f else 0
            right = files[f + 1] if f < 7 else 0
            if left == 0 and right == 0:
                score -= 8

    enemy_pawns = board.pieces(chess.PAWN, not color)
    for sq in pawns:
        f = chess.square_file(sq)
        rank = chess.square_rank(sq)
        rel_rank = rank if color == chess.WHITE else 7 - rank
        passed = True
        for esq in enemy_pawns:
            ef = chess.square_file(esq)
            if abs(ef - f) > 1:
                continue
            erank = chess.square_rank(esq)
            erel = erank if color == chess.WHITE else 7 - erank
            if erel > rel_rank:
                passed = False
                break
        if passed and rel_rank >= 2:
            score += 12 + 6 * rel_rank
    return score


def rook_files(board: chess.Board, color: chess.Color) -> int:
    own = board.pieces(chess.PAWN, color)
    enemy = board.pieces(chess.PAWN, not color)
    score = 0
    for sq in board.pieces(chess.ROOK, color):
        mask = chess.BB_FILES[chess.square_file(sq)]
        if not (own & mask):
            score += 12 if enemy & mask else 20
    return score


# Weight of each attacker type when it bears on a square in the king zone.
KING_ZONE_WEIGHT = {
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 5,
}

# Danger in centipawns for a given number of weighted attack units. Rises
# faster than linearly: two or three attackers on the king is much more
# than twice as dangerous as one, which a linear scale would understate.
KING_DANGER = (
    0, 0, 10, 25, 45, 70, 100, 135, 175, 220,
    270, 320, 370, 420, 470, 520, 570,
)


def king_zone(king_square: int) -> list[int]:
    """The king's own square plus its (up to 8) immediately adjacent ones."""
    kf = chess.square_file(king_square)
    kr = chess.square_rank(king_square)
    squares = []
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            f, r = kf + df, kr + dr
            if 0 <= f <= 7 and 0 <= r <= 7:
                squares.append(chess.square(f, r))
    return squares


def king_safety(board: chess.Board, color: chess.Color) -> int:
    """Danger score for `color`'s king from enemy pieces bearing on its
    zone. The caller scales this by game phase: a king caught in the open
    matters far less once most attacking material is off the board, but
    matters a great deal in the middlegame - the exact pattern behind the
    AlphaGambit loss (queen grabbed pawns, king had no cover, got mated
    while nominally ahead on material)."""
    king_sq = board.king(color)
    if king_sq is None:
        return 0

    enemy = not color
    units = 0
    for sq in king_zone(king_sq):
        for attacker_sq in board.attackers(enemy, sq):
            piece = board.piece_type_at(attacker_sq)
            if piece is not None:
                units += KING_ZONE_WEIGHT.get(piece, 0)

    return KING_DANGER[min(units, len(KING_DANGER) - 1)]


def evaluate(board: chess.Board) -> int:
    """Static score from the side-to-move perspective."""
    white = 0
    black = 0
    phase = 0

    for color in (chess.WHITE, chess.BLACK):
        total = 0
        for pt, pst in PSTS.items():
            bitboard = board.pieces(pt, color)
            value = VALUES[pt]
            if color == chess.WHITE:
                for sq in bitboard:
                    total += value + pst[sq]
            else:
                for sq in bitboard:
                    total += value + pst[chess.square_mirror(sq)]
            if pt != chess.PAWN:
                phase += value * len(bitboard)

        # Bishop pair.
        if len(board.pieces(chess.BISHOP, color)) >= 2:
            total += 30

        total += pawn_structure(board, color)
        total += rook_files(board, color)

        if color == chess.WHITE:
            white = total
        else:
            black = total

    score = white - black
    phase_ratio = min(1.0, phase / PHASE_MAX)

    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is not None:
        score += int(KING_MG[wk] * phase_ratio + KING_EG[wk] * (1 - phase_ratio))
    if bk is not None:
        msq = chess.square_mirror(bk)
        score -= int(KING_MG[msq] * phase_ratio + KING_EG[msq] * (1 - phase_ratio))

    # Dynamic king safety, middlegame only (skip the cost once most attacking
    # material is gone, since phase_ratio would scale it near zero anyway).
    if phase_ratio > 0.15:
        score -= int(king_safety(board, chess.WHITE) * phase_ratio)
        score += int(king_safety(board, chess.BLACK) * phase_ratio)

    # A small tempo term helps quiet equal positions without contaminating
    # repetition semantics.
    score += 8 if board.turn == chess.WHITE else -8
    return score if board.turn == chess.WHITE else -score


# ---------------------------------------------------------------------------
# Quiescence
# ---------------------------------------------------------------------------

def quiescence(
    board: chess.Board,
    alpha: int,
    beta: int,
    state: SearchState,
    ply: int,
) -> int:
    touch_clock(state)

    if board.halfmove_clock >= 100:
        return 0
    if board.is_insufficient_material():
        return 0
    if board.is_checkmate():
        return -MATE + ply

    in_check = board.is_check()
    if not in_check:
        stand = evaluate(board)
        if stand >= beta:
            return stand
        if stand > alpha:
            alpha = stand
    else:
        stand = -MATE

    if ply >= QMAX:
        return evaluate(board)

    if in_check:
        moves = list(board.legal_moves)
    else:
        moves = list(board.generate_legal_captures())
        # Checks are useful at the first two q-ply levels; beyond that the
        # capture-only tree is much cheaper and still resolves hanging pieces.
        if ply < 2:
            moves.extend(
                m
                for m in board.generate_legal_moves()
                if board.gives_check(m) and not board.is_capture(m)
            )

        if not moves:
            return alpha

        moves.sort(
            key=lambda m: move_score(board, m, None, min(ply, MAX_PLY - 1), None),
            reverse=True,
        )

    for move in moves:
        child_is_repeat = False
        board.push(move)
        try:
            key = pos_key(board)
            if key in state.path or GAME_REPETITIONS.get(key, 0) >= 2:
                child_is_repeat = True

            if child_is_repeat:
                score = 0
            else:
                state.path.add(key)
                try:
                    score = -quiescence(board, -beta, -alpha, state, ply + 1)
                finally:
                    state.path.discard(key)
        finally:
            board.pop()

        if score >= beta:
            return score
        if score > alpha:
            alpha = score
    return alpha


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------

def negamax(
    board: chess.Board,
    depth: int,
    alpha: int,
    beta: int,
    state: SearchState,
    ply: int,
    prev_move: chess.Move | None,
) -> int:
    touch_clock(state)

    if board.halfmove_clock >= 100:
        return 0
    if board.is_insufficient_material():
        return 0
    if board.is_checkmate():
        return -MATE + ply
    if depth <= 0:
        return quiescence(board, alpha, beta, state, ply)

    original_alpha = alpha
    key = tt_key(board)
    entry = TT.get(key)
    tt_move: chess.Move | None = None

    if entry is not None:
        tt_move = entry.move
        if entry.depth >= depth:
            cached = mate_from_tt(entry.score, ply)
            if entry.flag == TT_EXACT:
                return cached
            if entry.flag == TT_LOWER:
                alpha = max(alpha, cached)
            elif entry.flag == TT_UPPER:
                beta = min(beta, cached)
            if alpha >= beta:
                return cached

    in_check = board.is_check()

    # Check extension: a position where we're in check is tactically sharp
    # and easy to misjudge at a shallow horizon, so give it one extra ply of
    # search. Bounded by ply (not by a separate counter) so a long forced-
    # check sequence still terminates via the normal ply/time limits rather
    # than extending indefinitely.
    if in_check and ply < MAX_PLY - 4:
        depth += 1

    # Null move: a fail-high means the side to move has enough spare strength
    # that we can prune this subtree. Disabled in low-material / likely-zugzwang
    # positions and when already in check.
    non_pawns = (
        len(board.pieces(chess.KNIGHT, board.turn))
        + len(board.pieces(chess.BISHOP, board.turn))
        + len(board.pieces(chess.ROOK, board.turn))
        + len(board.pieces(chess.QUEEN, board.turn))
    )
    if (
        depth >= 5
        and ply > 0
        and not in_check
        and board.pieces(chess.PAWN, board.turn)
        and non_pawns >= 3
    ):
        board.push(chess.Move.null())
        try:
            nk = pos_key(board)
            repeated = nk in state.path
            if repeated:
                null_score = 0
            else:
                state.path.add(nk)
                try:
                    null_score = -negamax(
                        board,
                        depth - 1 - 2,
                        -beta,
                        -beta + 1,
                        state,
                        ply + 1,
                        chess.Move.null(),
                    )
                finally:
                    state.path.discard(nk)
        finally:
            board.pop()
        if null_score >= beta:
            return null_score

    moves = order_moves(board, tt_move, ply, prev_move)
    if not moves:
        return -MATE + ply if in_check else 0

    best_score = -MATE
    best_move: chess.Move | None = None
    quiet_tried: list[chess.Move] = []

    for index, move in enumerate(moves):
        touch_clock(state)

        quiet = is_quiet(board, move)
        tactical_check = board.gives_check(move)
        use_lmr = (
            index >= 4
            and depth >= 4
            and quiet
            and not tactical_check
            and not in_check
        )

        board.push(move)
        try:
            child_key = pos_key(board)
            repeated = (
                GAME_REPETITIONS.get(child_key, 0) >= 2
                or child_key in state.path
            )
            if repeated:
                score = 0
            else:
                state.path.add(child_key)
                try:
                    full_depth = depth - 1
                    search_depth = max(0, full_depth - 1) if use_lmr else full_depth

                    if index == 0:
                        score = -negamax(
                            board,
                            search_depth,
                            -beta,
                            -alpha,
                            state,
                            ply + 1,
                            move,
                        )
                    else:
                        # PVS scout search.
                        score = -negamax(
                            board,
                            search_depth,
                            -alpha - 1,
                            -alpha,
                            state,
                            ply + 1,
                            move,
                        )

                        # A reduced/scout success must be confirmed at full
                        # depth. This is the critical LMR correctness rule.
                        if score > alpha:
                            score = -negamax(
                                board,
                                full_depth,
                                -beta,
                                -alpha,
                                state,
                                ply + 1,
                                move,
                            )
                finally:
                    state.path.discard(child_key)
        finally:
            board.pop()

        if score > best_score:
            best_score = score
            best_move = move

        if score > alpha:
            alpha = score

        if alpha >= beta:
            if quiet:
                store_killer(ply, move)
                update_history(move, depth, True)
                if prev_move is not None:
                    COUNTERMOVE[hist_index(prev_move)] = move
                for old in quiet_tried:
                    update_history(old, depth, False)
            break

        if quiet:
            quiet_tried.append(move)

    if best_move is not None:
        if len(TT) >= TT_MAX:
            TT.clear()
        if best_score <= original_alpha:
            flag = TT_UPPER
        elif best_score >= beta:
            flag = TT_LOWER
        else:
            flag = TT_EXACT
        TT[key] = TTEntry(depth, mate_to_tt(best_score, ply), flag, best_move)

    return best_score


# ---------------------------------------------------------------------------
# Root search / time management
# ---------------------------------------------------------------------------

def root_search(
    board: chess.Board,
    depth: int,
    alpha: int,
    beta: int,
    state: SearchState,
    previous_best: chess.Move | None,
) -> tuple[chess.Move | None, int]:
    root = tt_key(board)
    entry = TT.get(root)
    tt_move = entry.move if entry is not None else None
    moves = order_moves(board, tt_move, 0, None)

    if previous_best is not None and previous_best in moves:
        moves.remove(previous_best)
        moves.insert(0, previous_best)

    best: chess.Move | None = None
    best_score = -MATE
    state.path.add(pos_key(board))

    for index, move in enumerate(moves):
        if time.perf_counter() >= state.deadline:
            raise SearchTimeout

        # Never actively select a move that is already a known third occurrence.
        board.push(move)
        try:
            child_key = pos_key(board)
            repeated = (
                GAME_REPETITIONS.get(child_key, 0) >= 2
                or child_key in state.path
            )
            if repeated:
                score = 0
            else:
                state.path.add(child_key)
                try:
                    if index == 0:
                        score = -negamax(
                            board, depth - 1, -beta, -alpha, state, 1, move
                        )
                    else:
                        score = -negamax(
                            board, depth - 1, -alpha - 1, -alpha, state, 1, move
                        )
                        if score > alpha:
                            score = -negamax(
                                board, depth - 1, -beta, -alpha, state, 1, move
                            )
                finally:
                    state.path.discard(child_key)
        finally:
            board.pop()

        if score > best_score:
            best_score = score
            best = move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break

    return best, best_score


def budget_ms(time_left_ms: int, urgent: bool = False) -> int:
    # The old implementation hard-capped itself at 5s even with a huge clock.
    # This engine uses a fraction of the remaining clock, while preserving a
    # hard reserve so an unfinished iteration can never consume the position.
    if time_left_ms <= 200:
        return max(30, time_left_ms - 20)

    target = time_left_ms // 16 + 350

    # Tactically loud positions (we're in check, or the position on the
    # board right now was reached by a capture) get extra time: this is
    # exactly the situation where a shallow search misses a piece a couple
    # of moves deep, which is what cost the Round 15 loss - the engine had
    # ~1.8s at a critical moment and never looked far enough to see Qxe8+
    # coming after Qxb2.
    if urgent:
        target = int(target * 1.6)

    target = min(MOVE_CAP_MS, max(MIN_BUDGET_MS, target))
    target = min(target, max(MIN_BUDGET_MS, time_left_ms - RESERVE_MS))
    return target


def record_game_position(key: Any) -> None:
    GAME_REPETITIONS[key] = GAME_REPETITIONS.get(key, 0) + 1


def any_piece_attacked(board: chess.Board, color: chess.Color) -> bool:
    """True if any of `color`'s own pieces are currently attacked by the
    enemy. Catches threats a capture/check check misses entirely - e.g. a
    quiet queen move that lines up on an undefended rook, exactly the shape
    of the Round 15 loss (Qf7 attacked the rook on e8 with no capture and
    no check involved)."""
    enemy = not color
    for square, piece in board.piece_map().items():
        if piece.color == color and board.is_attacked_by(enemy, square):
            return True
    return False


def emergency_move(board: chess.Board) -> chess.Move:
    moves = order_moves(board, None, 0, None)
    return moves[0] if moves else chess.Move.null()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if not legal:
        return "0000"

    root_key = pos_key(board)
    record_game_position(root_key)

    if len(legal) == 1:
        move = legal[0]
        board.push(move)
        try:
            record_game_position(pos_key(board))
        finally:
            board.pop()
        return move.uci()

    # "Urgent" = we're in check, the position arose from a capture, or any
    # of our own pieces are currently attacked - the last case is what the
    # Round 15 loss actually needed: Qf7 threatened the rook on e8 without
    # capturing anything or giving check, so neither of the first two
    # signals fired there.
    urgent = (
        board.is_check()
        or board.halfmove_clock == 0
        or any_piece_attacked(board, board.turn)
    )

    budget = budget_ms(time_left_ms, urgent)
    deadline = time.perf_counter() + budget / 1000.0
    state = SearchState(
        deadline=deadline,
        check_mask=FAST_CHECK_MASK if budget < 700 else NORMAL_CHECK_MASK,
        path={root_key},
    )

    # Deterministic legal fallback before any deep search.
    best = emergency_move(board)
    best_score = -MATE
    previous_score: int | None = None

    for depth in range(1, MAX_DEPTH + 1):
        try:
            # Aspiration windows after the first completed iteration.
            if previous_score is not None and depth >= 4:
                window = 50
                alpha = max(-MATE, previous_score - window)
                beta = min(MATE, previous_score + window)
                while True:
                    candidate, score = root_search(
                        board, depth, alpha, beta, state, best
                    )
                    if score <= alpha:
                        window *= 2
                        alpha = max(-MATE, previous_score - window)
                        if time.perf_counter() >= state.deadline:
                            raise SearchTimeout
                        continue
                    if score >= beta:
                        window *= 2
                        beta = min(MATE, previous_score + window)
                        if time.perf_counter() >= state.deadline:
                            raise SearchTimeout
                        continue
                    break
            else:
                candidate, score = root_search(
                    board, depth, -MATE, MATE, state, best
                )
        except SearchTimeout:
            break

        if candidate is not None and candidate in legal:
            # Only a *completed* iteration commits the move.
            best = candidate
            best_score = score
            previous_score = score

        if abs(best_score) >= MATE_BOUND:
            break
        if time.perf_counter() + 0.010 >= deadline:
            break

    if best not in legal:
        best = emergency_move(board)

    board.push(best)
    try:
        record_game_position(pos_key(board))
    finally:
        board.pop()
    return best.uci()
