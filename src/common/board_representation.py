import numpy as np
import chess
import torch

piece_channel = {chess.KING: 0, chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 3, chess.KNIGHT: 5, chess.PAWN: 5}

def board_to_numpy(board: chess.Board, dim=0, dtype=np.float16):
    tensor = np.zeros((13, 8, 8), dtype=dtype)

    # piece_types = [chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]

    # for idx, piece_type in enumerate(piece_types):
    #     _pieces_tensor(board, tensor[idx], piece_type)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank, file = 7 - chess.square_rank(square), chess.square_file(square)
            if piece.color == chess.WHITE:
                tensor[piece_channel[piece.piece_type], rank, file] = 1
            else:
                tensor[piece_channel[piece.piece_type], rank, file] = -1

    white_castle_king_side = int(board.has_kingside_castling_rights(chess.WHITE))
    white_castle_queen_side = int(board.has_queenside_castling_rights(chess.WHITE))
    black_castle_king_side = -int(board.has_kingside_castling_rights(chess.BLACK))
    black_castle_queen_side = -int(board.has_queenside_castling_rights(chess.BLACK))

    en_passant_square = board.ep_square
    if en_passant_square:
        if board.turn == chess.WHITE:
            tensor[10][_square_to_tensor_indices(en_passant_square)] = 1
        else:
            tensor[11][_square_to_tensor_indices(en_passant_square)] = -1

    tensor[6] = np.full((8, 8), white_castle_king_side, dtype=dtype)
    tensor[7] = np.full((8, 8), white_castle_queen_side, dtype=dtype)
    tensor[8] = np.full((8, 8), black_castle_king_side, dtype=dtype)
    tensor[9] = np.full((8, 8), black_castle_queen_side, dtype=dtype)
    tensor[12] = np.full((8, 8), 1 if board.turn == chess.WHITE else -1, dtype=dtype)

    return np.moveaxis(tensor, 0, dim)


def _square_to_tensor_indices(square):
    rank, file = 7 - chess.square_rank(square), chess.square_file(square)
    return rank, file


def _pieces_tensor(board, tensor, piece_type):
    white_pieces_squares = board.pieces(piece_type, chess.WHITE)
    black_pieces_squares = board.pieces(piece_type, chess.BLACK)

    for square in white_pieces_squares:
        rank, file = _square_to_tensor_indices(square)
        tensor[rank][file] = 1

    for square in black_pieces_squares:
        rank, file = _square_to_tensor_indices(square)
        tensor[rank][file] = -1


def move_to_numpy(board: str | chess.Board, move: str | chess.Move, dim=0):

    if isinstance(board, str):
        board = chess.Board(board)
    else:
        board = board.copy()

    pre_move_tensor = board_to_numpy(board, dim)

    if isinstance(move, str):
        move = chess.Move.from_uci(move)

    board.push(move)
    post_move_tensor = board_to_numpy(board, dim)

    move_tensor = np.concatenate([pre_move_tensor, post_move_tensor], axis=dim)

    return move_tensor


def move_to_tensor(position_fen: str | chess.Board, move: str | chess.Move, dim=0, dtype=torch.float16):
    move_numpy = move_to_numpy(position_fen, move, dim=dim)

    return torch.from_numpy(move_numpy).to(dtype)