from sympy import ordered
import torch
import torch.nn as nn
import chess
import board_representation.board_representation_2 as br
import numpy as np
import random
from more_itertools import partition
import board_representation.sentimate as br_sentimate

class Agent:
    def __init__(self) -> None:
        pass

    def play(self, board: chess.Board):
        pass

class ModelAgent(Agent):
    def __init__(self, model, convert_fn = br.move_to_tensor, min_for_black = False) -> None:
        super().__init__()
        self.model = model
        self.model_device = next(iter(model.parameters())).device
        self.model_dtype = next(iter(model.parameters())).dtype
        self.convert_fn = convert_fn
        self.min_for_black = min_for_black

    def batch_legal_moves(self, board: chess.Board):
        move_tensors = [self.convert_fn(board.fen(), move) for move in board.legal_moves]
        return torch.stack(move_tensors)

    def play(self, board: chess.Board):
        legal_moves = list(board.legal_moves)

        if len(legal_moves) == 0:
            return None
        
        batch_moves = self.batch_legal_moves(board).to(self.model_device, dtype=self.model_dtype)

        self.model.eval()
        with torch.inference_mode():
            out = self.model(batch_moves)

            if out.shape[-1] == 2:
                out = out[:, 1]

            best_move_ind = out.argmax().item()
            
            best_move = list(board.legal_moves)[best_move_ind]

            return best_move

        
class RandomAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    def play(self, board: chess.Board):
        legal_moves = list(board.legal_moves)

        if len(legal_moves) == 0:
            return None

        random_move = random.choice(legal_moves)

        return random_move



class NegaMaxAgent(Agent):
    def __init__(self, depth = 3) -> None:
        super().__init__()
        self.depth = depth

    def play(self, board: chess.Board):
        best_score = -float('inf')
        best_moves = []
        alpha = -float('inf')
        beta = float('inf')
        
        for move in self.ordered_moves(board):
            board.push(move)
            score = -self.negamax(board, self.depth - 1, -beta, -alpha)
            board.pop()
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)
            alpha = max(alpha, score)
        
        return self.choose_from_best_moves(board, best_moves)


    def negamax(self, board, depth, alpha, beta):
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)
        
        max_score = -float('inf')
        for move in self.ordered_moves(board):
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            max_score = max(max_score, score)
            alpha = max(alpha, score)
            if alpha > beta:
                break
        return max_score
    

    def ordered_moves(self, board):
        return board.legal_moves
    
    def choose_from_best_moves(self, board, best_moves):
        pass
    
    # Evaluate the board from the perspective of the current player
    def evaluate_board(self, board):
        pass


class NegaMaxMaterialAgent(NegaMaxAgent):
    def __init__(self, depth = 3) -> None:
        super().__init__(depth)
    
    def evaluate_board(self, board):
        color = 1 if board.turn == chess.WHITE else -1
        return color * material_difference(board)
    
    def choose_from_best_moves(self, board, best_moves):
        return random.choice(best_moves)
    
def move_to_tensor(board, move):
    pl = br_sentimate.move_to_piece_list(board, move)
    array = br_sentimate.piece_lists_to_board_array_only_pieces(*pl)
    tensor = torch.from_numpy(array)
    return tensor

class NegaMaxMaterialModelAgent(NegaMaxMaterialAgent):
    def __init__(self, model, convert_fn = move_to_tensor, depth=3) -> None:
        super().__init__(depth)
        self.model = model
        self.convert_fn = convert_fn

    def choose_from_best_moves(self, board, best_moves):
        batch_moves = [self.convert_fn(board.fen(), move) for move in best_moves]
        batch_moves = torch.stack(batch_moves)

        self.model.eval()
        with torch.inference_mode():
            out = self.model(batch_moves)

            if out.shape[-1] == 2:
                out = out[:, 1]

            best_move_ind = out.argmax().item()
            
            best_move = best_moves[best_move_ind]

            return best_move
        

class NegaMaxModelSearchAgent(Agent):
    def __init__(self, model, convert_fn = move_to_tensor, depth = 3) -> None:
        super().__init__()
        self.depth = depth
        # Model should return a probability of move being good
        self.model = model
        self.convert_fn = convert_fn

    def play(self, board: chess.Board):
        best_score = -float('inf')
        best_moves = []
        alpha = -float('inf')
        beta = float('inf')
        
        ordered_moves, move_scores = self.ordered_moves_and_scores(board)

        # print("ORDERED MOVES:", ordered_moves)
        # print("LEGAL MOVES:", list(board.legal_moves))

        for move, move_score in zip(ordered_moves, move_scores):
            board.push(move)
            score = move_score.item() -self.negamax(board, self.depth - 1, -beta, -alpha, False)
            board.pop()
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)
            alpha = max(alpha, score)

        return self.choose_from_best_moves(board, best_moves)


    def negamax(self, board, depth, alpha, beta, max_player):
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)
        
        max_score = -float('inf')

        ordered_moves, move_scores = self.ordered_moves_and_scores(board)

        # print("ORDERED MOVES:", ordered_moves)
        # print("LEGAL MOVES:", list(board.legal_moves))

        for move, move_score in zip(ordered_moves, move_scores):
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, not max_player)
            if max_player:
                score += move_score.item()
            board.pop()
            max_score = max(max_score, score)
            alpha = max(alpha, score)
            if alpha > beta:
                break
        return max_score
    

    def ordered_moves_and_scores(self, board: chess.Board):
        legal_moves = np.array(list(board.legal_moves))
        model_input = [self.convert_fn(board.fen(), move) for move in board.legal_moves]
        
        with torch.inference_mode():
            model_input = torch.stack(model_input)
            move_probabilites = self.model(model_input)
            # move_probabilites = torch.log(move_probabilites)

        sorted_indices = move_probabilites.argsort(descending=True)

        ordered_moves = legal_moves[sorted_indices]

        if len(legal_moves) == 1:
            ordered_moves = legal_moves

        scores = move_probabilites[sorted_indices]

        return ordered_moves, scores
    
    def choose_from_best_moves(self, board, best_moves):
        return best_moves[0]
    
    # Evaluate the board from the perspective of the current player
    def evaluate_board(self, board):
        return 0



def material(board):
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    white_material = (
        chess.popcount(white & board.pawns) +
        3 * chess.popcount(white & board.knights) +
        3 * chess.popcount(white & board.bishops) +
        5 * chess.popcount(white & board.rooks) +
        9 * chess.popcount(white & board.queens)
    )

    black_material = (
        chess.popcount(black & board.pawns) +
        3 * chess.popcount(black & board.knights) +
        3 * chess.popcount(black & board.bishops) +
        5 * chess.popcount(black & board.rooks) +
        9 * chess.popcount(black & board.queens)
    )

    return white_material, black_material

def material_difference(board):
    if board.is_checkmate():
        return -float('inf') if board.turn == chess.WHITE else float('inf')
    white_material, black_material = material(board)
    return white_material - black_material
