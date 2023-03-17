import chess
import torch
import numpy as np


def board_to_tensor(board: chess.Board, dim=0):
    # KINGS - 1 layer
    kings = _pieces_tensor(board, chess.KING)

    # QUEENS - 1 layer
    queens = _pieces_tensor(board, chess.QUEEN)

    # ROOKS - 1 layer
    rooks = _pieces_tensor(board, chess.ROOK)

    # BISHOPS - 1 layer
    bishops = _pieces_tensor(board, chess.BISHOP)

    # KNIGTH - 1 layer
    knights = _pieces_tensor(board, chess.KNIGHT)

    # PAWNS - 1 layer
    pawns = _pieces_tensor(board, chess.PAWN)

    # CASTLING RIGHTS - 4 layers
    white_castle_king_side = torch.zeros((8,8)) + int(board.has_kingside_castling_rights(chess.WHITE))
    white_castle_queen_side = torch.zeros((8,8)) + int(board.has_queenside_castling_rights(chess.WHITE))

    black_castle_king_side = torch.zeros((8,8)) - int(board.has_kingside_castling_rights(chess.BLACK))
    black_castle_queen_side = torch.zeros((8,8)) - int(board.has_queenside_castling_rights(chess.BLACK))

    # EN PASSANT - 2 layers
    en_passant_white = torch.zeros((8,8))
    en_passant_black = torch.zeros((8,8))
    if board.ep_square:
        if board.turn == chess.WHITE:
            en_passant_white[_square_to_tensor_indices(board.ep_square)] = 1
        else:
            en_passant_black[_square_to_tensor_indices(board.ep_square)] = -1

    # WHICH MOVE - 1 layer
    which_move = torch.zeros((8,8)) + (1 if board.turn == chess.WHITE else -1)

    layers_list = [kings, queens, rooks, bishops, knights, pawns, white_castle_king_side, white_castle_queen_side,
                   black_castle_king_side, black_castle_queen_side, en_passant_white, en_passant_black, which_move]

    return torch.stack(layers_list, dim=dim)


def _square_to_tensor_indices(square):
    rank, file = 7-chess.square_rank(square), chess.square_file(square)
    return rank, file

def _pieces_tensor(board, piece_type):
    tensor = torch.zeros((8, 8))
    
    white_pieces_squares = board.pieces(piece_type, chess.WHITE)

    for square in white_pieces_squares:
        rank, file = _square_to_tensor_indices(square)
        tensor[rank][file] = 1


    black_pieces_squares = board.pieces(piece_type, chess.BLACK)

    for square in black_pieces_squares:
        rank, file = _square_to_tensor_indices(square)
        tensor[rank][file] = -1

    return tensor


def move_to_tensor(position_fen: str, move_uci: str, dim = 0):
    board = chess.Board(position_fen)
    pre_move_tensor = board_to_tensor(board, dim)

    board.push_uci(move_uci)
    post_move_tensor = board_to_tensor(board, dim)

    move_tensor = torch.concat([pre_move_tensor, post_move_tensor], dim=dim)

    return move_tensor