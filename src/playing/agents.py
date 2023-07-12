import torch
import torch.nn as nn
import chess
import common.board_representation_2 as br
import numpy as np
import random
from more_itertools import partition

class Agent:
    def __init__(self) -> None:
        pass

    def play(self, board: chess.Board):
        pass

class ModelAgent(Agent):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.model_device = next(iter(model.parameters())).device
        self.model_dtype = next(iter(model.parameters())).dtype

    def batch_legal_moves(self, board: chess.Board):
        move_tensors = [br.move_to_tensor(board.fen(), move) for move in board.legal_moves]
        return torch.stack(move_tensors)

    def play(self, board: chess.Board):
        legal_moves = list(board.legal_moves)

        if len(legal_moves) == 0:
            return None
        
        batch_moves = self.batch_legal_moves(board).to(self.model_device, dtype=self.model_dtype)

        self.model.eval()
        with torch.inference_mode():
            out = self.model(batch_moves)
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



class MinimaxAgent(Agent):
    def __init__(self, model: nn.Module, max_depth, min_coef = 0.5, decay = 0.5) -> None:
        super().__init__()
        self.model = model
        self.model.cuda()
        self.model.eval()
        self.model_dtype = next(iter(model.parameters())).dtype
        self.max_depth = max_depth
        self.move_scores = {}
        self.min_coef = min_coef
        self.decay = decay
    

    def play(self, board: chess.Board):
        best_move = None
        max_eval = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        legal_moves_scores = self.eval_legal_moves(board)

        for move, move_score in legal_moves_scores:
            board.push(move)
            eval_score = move_score + self.decay * self.minimax_alpha_beta(board, self.max_depth - 1, alpha, beta, False)
            board.pop()
            if eval_score > max_eval or best_move == None:
                max_eval = eval_score
                best_move = move
        return best_move

    def minimax_alpha_beta(self, board: chess.Board, depth, alpha, beta, maximizing_player):
        if board.is_game_over():
            if board.result() == '1/2-1/2':
                return 0
            elif maximizing_player:
                return float('-inf')
            else:
                return float('inf')

        if depth == 0:
            return 0

        legal_moves_scores = self.eval_legal_moves(board)

        if maximizing_player:
            max_eval = float('-inf')
            for move, move_score in legal_moves_scores:
                board.push(move)
                eval_score = move_score + self.decay * self.minimax_alpha_beta(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move, move_score in legal_moves_scores:
                board.push(move)
                eval_score = -self.min_coef*move_score + self.decay * self.minimax_alpha_beta(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    def move_eval(self, board, move):
        epd = board.epd()
        key = epd + move.uci()

        if key in self.move_scores:
            return self.move_scores[key]
        
        move_tensor = br.move_to_tensor(board, move, dtype=self.model_dtype).unsqueeze(0).cuda()

        with torch.inference_mode():
            move_score = self.model(move_tensor).item()

            self.move_scores[key] = move_score

            return move_score

    def batch_legal_moves(self, board: chess.Board):
        move_tensors = [br.move_to_tensor(board, move) for move in board.legal_moves]
        return torch.stack(move_tensors)

    def eval_legal_moves(self, board: chess.Board):
        legal_moves = np.array(list(board.legal_moves))
        batch_moves = self.batch_legal_moves(board).cuda()

        with torch.inference_mode():
            move_scores = self.model(batch_moves).flatten()
            argsort_scores = torch.argsort(move_scores, descending=True).cpu().numpy()
            sorted_scores = move_scores[argsort_scores].cpu().numpy()
            sorted_moves = legal_moves[argsort_scores]
            
            return zip(sorted_moves, sorted_scores)



    def position_eval(self, move_score, position_score, position_result):
        return 0



def move_key(board: chess.Board, move: chess.Move):
    return (board.epd(), move.uci())



    