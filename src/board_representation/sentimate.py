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
            piece = board.piece_at(chess.square(j, i))
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



def board_to_piece_list(board: chess.Board):
    white_pieces = []
    black_pieces = []
    turn = 1 if board.turn == chess.WHITE else -1

    for i in range(8):
        for j in range(8):
            piece = board.piece_at(chess.square(j, i)) 
            if piece is not None:
                coord = (piece.piece_type - 1) * 64 + i*8 + j
                if piece.color == chess.WHITE:
                    white_pieces.append(coord)
                else:
                    black_pieces.append(coord)

    # return np.array(white_pieces, dtype=np.int8), np.array(black_pieces, dtype=np.int8)
    return np.array(white_pieces, dtype=np.int16), np.array(black_pieces, dtype=np.int16), np.array([turn], dtype=np.int16)

def move_to_piece_list(board, move):
    if isinstance(move, str):
        move = chess.Move.from_uci(move)
    if isinstance(board, str):
        board = chess.Board(board)

    pre_move = board_to_piece_list(board)
    board.push(move)
    post_move = board_to_piece_list(board)
    board.pop()
    return *pre_move, *post_move


def piece_lists_to_board_array(pre_move_white, pre_move_black, pre_move_turn, post_move_white, post_move_black, post_move_turn):
    move_array = np.zeros((26 * 64), dtype=np.float32)
    move_array[pre_move_white] = 1
    move_array[6*64:][pre_move_black] = -1
    move_array[13*64:][post_move_white] = 1
    move_array[19*64:][post_move_black] = -1
    move_array = move_array.reshape(-1,8,8)
    move_array[12] = pre_move_turn
    move_array[25] = post_move_turn

    return move_array


def piece_lists_to_board_array_zero_move(pre_move_white, pre_move_black, pre_move_turn, post_move_white, post_move_black, post_move_turn):
    move_array = np.zeros((26 * 64), dtype=np.float32)
    move_array[pre_move_white] = 1
    move_array[6*64:][pre_move_black] = -1
    move_array[13*64:][post_move_white] = 1
    move_array[19*64:][post_move_black] = -1
    move_array = move_array.reshape(-1,8,8)
    # move_array[12] = pre_move_turn
    # move_array[25] = post_move_turn

    return move_array


def piece_lists_to_board_array_only_pieces(pre_move_white, pre_move_black, pre_move_turn, post_move_white, post_move_black, post_move_turn):
    move_array = np.zeros((24 * 64), dtype=np.float32)
    move_array[pre_move_white] = 1
    move_array[6*64:][pre_move_black] = -1
    move_array[12*64:][post_move_white] = 1
    move_array[18*64:][post_move_black] = -1
    move_array = move_array.reshape(-1,8,8)

    return move_array