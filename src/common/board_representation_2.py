import chess
import numpy as np

import torch


def get_chess_representation(board, move):
    # Create an empty 26x8x8 representation
    representation = np.zeros((26, 8, 8), dtype=np.float16)

    # Define piece mapping for indexing the channels
    piece_mapping = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    # Iterate over each square on the board
    for rank in range(8):
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)

            if piece is not None:
                # Get the channel index based on the piece type
                piece_index = piece_mapping[piece.piece_type]

                # Set +1 for white pieces, -1 for black pieces
                if piece.color == chess.WHITE:
                    representation[piece_index, rank, file] = 1
                else:
                    representation[piece_index, rank, file] = -1

    # Set the channel for the side to move
    representation[12, :, :] = -1 if board.turn == chess.BLACK else 1

    # Set the channels for castle rights
    castle_rights = board.castling_rights
    if castle_rights & chess.BB_H1:
        representation[6, :, :] = 1  # White kingside
    if castle_rights & chess.BB_A1:
        representation[7, :, :] = 1  # White queenside
    if castle_rights & chess.BB_H8:
        representation[8, :, :] = -1  # Black kingside
    if castle_rights & chess.BB_A8:
        representation[9, :, :] = -1  # Black queenside

    # Set the channel for en passant possibility
    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square)
        ep_rank = chess.square_rank(board.ep_square)
        representation[10, ep_rank, ep_file] = 1

    # Make the move on a temporary board to get the new position
    temp_board = board.copy()
    temp_board.push(move)

    # Repeat the process for the new position
    for rank in range(8):
        for file in range(8):
            square = chess.square(file, rank)
            piece = temp_board.piece_at(square)

            if piece is not None:
                piece_index = piece_mapping[piece.piece_type]

                if piece.color == chess.WHITE:
                    representation[piece_index + 13, rank, file] = 1
                else:
                    representation[piece_index + 13, rank, file] = -1

    # Set the channels for castle rights in the new position
    new_castle_rights = temp_board.castling_rights
    if new_castle_rights & chess.BB_H1:
        representation[11, :, :] = 1  # White kingside
    if new_castle_rights & chess.BB_A1:
        representation[12, :, :] = 1  # White queenside
    if new_castle_rights & chess.BB_H8:
        representation[13, :, :] = -1  # Black kingside
    if new_castle_rights & chess.BB_A8:
        representation[14, :, :] = -1  # Black queenside

    # Set the channel for en passant possibility in the new position
    if temp_board.ep_square is not None:
        ep_file = chess.square_file(temp_board.ep_square)
        ep_rank = chess.square_rank(temp_board.ep_square)
        representation[15, ep_rank, ep_file] = 1

    return representation


def board_to_numpy(board: chess.Board):
    # Create an empty 26x8x8 representation
    representation = np.zeros((13, 8, 8), dtype=np.float16)

    # Define piece mapping for indexing the channels
    piece_mapping = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    # Iterate over each square on the board
    for rank in range(8):
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)

            if piece is not None:
                # Get the channel index based on the piece type
                piece_index = piece_mapping[piece.piece_type]

                # Set +1 for white pieces, -1 for black pieces
                if piece.color == chess.WHITE:
                    representation[piece_index, rank, file] = 1
                else:
                    representation[piece_index, rank, file] = -1

    # Set the channel for the side to move
    representation[12, :, :] = -1 if board.turn == chess.BLACK else 1

    # Set the channels for castle rights
    castle_rights = board.castling_rights
    if castle_rights & chess.BB_H1:
        representation[6, :, :] = 1  # White kingside
    if castle_rights & chess.BB_A1:
        representation[7, :, :] = 1  # White queenside
    if castle_rights & chess.BB_H8:
        representation[8, :, :] = -1  # Black kingside
    if castle_rights & chess.BB_A8:
        representation[9, :, :] = -1  # Black queenside

    # Set the channel for en passant possibility
    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square)
        ep_rank = chess.square_rank(board.ep_square)
        representation[10, ep_rank, ep_file] = 1

    return representation



def move_to_tensor(board, move, dtype=torch.float16):
    if isinstance(board, str):
        board = chess.Board(board)

    if isinstance(move, str):
        move = chess.Move.from_uci(move)
        
    representation = get_chess_representation(board, move)
    representation_tensor = torch.from_numpy(representation).to(dtype)

    return representation_tensor


def next_positions(board: chess.Board, moves = None):
    ret = []
    
    if not moves:
        moves = board.legal_moves

    for move in moves:
        new_board = board.copy()
        new_board.push(move)
        ret.append(new_board)

    return ret