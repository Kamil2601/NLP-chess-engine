import chess
import numpy as np
import torch

def board_to_input(board, array=None):
    piece_to_index = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
                      'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10, 'K': 11}

    # Initialize the 8x8x12 representation
    if array is None:
        input_representation = np.zeros((13, 8, 8))
    else:
        input_representation = array

    # Mapping of piece characters to their indices in the piece_to_index dictionary
    for i in range(8):
        for j in range(8):
            piece = board.piece_at(chess.square(j, 7 - i))  # Flipping the board to match the description
            if piece is not None:
                input_representation[piece_to_index[piece.symbol()], i, j] = 1 if piece.color == chess.WHITE else -1

    # Adding a channel to signify whose turn it is (all +1 for white, all -1 for black)
    input_representation[12] = 1 if board.turn == chess.WHITE else -1

    return input_representation

def move_to_numpy(board, move, dtype=np.float32):
    if isinstance(board, str):
        board = chess.Board(board)

    if isinstance(move, str):
        move = chess.Move.from_uci(move)

    # Make the move on a copy of the board to get the pre- and post-move representations
    board_copy = board.copy()
    board_copy.push(move)

    final_representation = np.zeros((26, 8, 8), dtype=dtype)

    pre_move_representation = board_to_input(board, array=final_representation)
    post_move_representation = board_to_input(board_copy, array=final_representation[13:])

    # Stack the two representations to get the final 8x8x26 representation
    # final_representation = np.concatenate((pre_move_representation, post_move_representation), axis=2)

    return final_representation


def move_to_tensor(board, move, dtype=torch.float16):
    return torch.from_numpy(move_to_numpy(board, move)).to(dtype=dtype)