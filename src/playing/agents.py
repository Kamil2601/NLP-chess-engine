import torch
import torch.nn as nn
import chess
import board_representation.board_representation_2 as br
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

    def batch_legal_moves(self, board: chess.Board):
        move_tensors = [br.move_to_tensor(board, move, self.model_dtype) for move in board.legal_moves]
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




    
class FasterMinimaxAgent(MinimaxAgent):
    def __init__(self, model: nn.Module, max_depth, min_coef = 0.5, decay = 0.5):
        super().__init__(model, max_depth, min_coef, decay)
        self.board_numpy_cache = {}

    def batch_legal_moves(self, board: chess.Board):
        next_pos = br.next_positions(board)
        board_np = self.board_numpy(board)
        next_board_nps = list(map(self.board_numpy, next_pos))
        batch_list = [np.concatenate([board_np, next_np]) for next_np in next_board_nps]
        batch_numpy = np.stack(batch_list)
        batch_tensor = torch.from_numpy(batch_numpy).to(self.model_dtype)

        return batch_tensor

    def board_numpy(self, board: chess.Board):
        # epd = board.epd()
        # res = self.board_numpy_cache.get(epd)

        # if not np.any(res):
        #     res = br.board_to_numpy(board)
        #     self.board_numpy_cache[epd] = res

        # return res

        return br.board_to_numpy(board)

    

piece_value_pos_eval = {
    'P': 1, 'p': -1,
    'N': 3, 'n': -3,
    'B': 3, 'b': -3,
    'R': 5, 'r': -5,
    'Q': 9, 'q': -9,
    'K': 0, 'k': 0,
}

piece_value = {
    chess.PAWN: 1,
    chess.BISHOP: 3,
    chess.KNIGHT: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
    None: 0,
}

def material_eval(board: chess.Board):
    pieces = map(str, board.piece_map().values())
    return sum([piece_value_pos_eval[p] for p in pieces])


class MiniMaxPositionEvalAgent(Agent):
    def __init__(self, max_depth):
        super().__init__()
        self.max_depth = max_depth

    def play(self, board: chess.Board):
        best_moves = []
        max_eval = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        possible_moves = self.order_legal_moves(board)

        for move in possible_moves:
            board.push(move)
            eval_score = self.minimax_alpha_beta(board, self.max_depth - 1, alpha, beta, False)
            board.pop()
            if eval_score > max_eval or best_moves == []:
                max_eval = eval_score
                best_moves = [move]
            elif eval_score == max_eval:
                best_moves.append(move)
                
        # print(list(map(str, best_moves)))
                
        return self.chose_from_best_moves(board, best_moves)
    
    
    def game_over_result(self, board: chess.Board, maximizing_player):
        if board.result() == '1/2-1/2':
                return 0
        elif maximizing_player:
            return float('-inf')
        else:
            return float('inf')
        
    def eval_position(self, board: chess.Board, maximizing_player):
        material_difference = material_eval(board)
        
        if board.turn == maximizing_player:
            return material_difference
        else:
            return -material_difference
    
    def chose_from_best_moves(self, board: chess.Board, best_moves):
        return np.random.choice(best_moves)
    

    def order_legal_moves(self, board: chess.Board, only_captures = False, square = None):
        if only_captures:
            if square:
                possible_moves = [move for move in board.legal_moves if board.is_capture(move) and move.to_square == square]
            else:
                possible_moves = [move for move in board.legal_moves if board.is_capture(move)]
        else:
            possible_moves = list(board.legal_moves)
            
        move_scores = []
        
        def move_score(move):
            res = 0
            
            if board.is_capture(move):
                res = piece_value[board.piece_type_at(move.to_square)]
                
            if move.promotion:
                res += piece_value[move.promotion]
                
            return res
            
        return sorted(possible_moves, key=move_score, reverse=True)


    def minimax_alpha_beta(self, board: chess.Board, depth, alpha, beta, maximizing_player):       
        if board.is_game_over():
            return self.game_over_result(board, maximizing_player)

        if depth == 0:
            if board.move_stack:
                return self.search_captures(board, alpha, beta, maximizing_player, square=board.peek().to_square)
            return self.eval_position(board, maximizing_player)
        
        possible_moves = self.order_legal_moves(board)


        if maximizing_player:
            max_eval = float('-inf')
            for move in possible_moves:
                board.push(move)
                eval_score = self.minimax_alpha_beta(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta < alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in possible_moves:
                board.push(move)
                eval_score = self.minimax_alpha_beta(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta < alpha:
                    break
            return min_eval
        
    def search_captures(self, board: chess.Board, alpha, beta, maximizing_player, square = None):       
        if board.is_game_over():
            return self.game_over_result(board, maximizing_player)
        
        captures = self.order_legal_moves(board, only_captures=True, square = square)
        
        if not captures:
            return self.eval_position(board, maximizing_player)
        
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in captures:
                board.push(move)
                eval_score = self.search_captures(board, alpha, beta, False, square=square)
                board.pop()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in captures:
                board.push(move)
                eval_score = self.search_captures(board, alpha, beta, True, square=square)
                board.pop()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval
        

class MiniMaxPositionEvalModelAgent(MiniMaxPositionEvalAgent):
    def __init__(self, model: nn.Module, max_depth):
        super().__init__(max_depth)
        self.model = model
        self.model.cuda()
        self.model.eval()
        self.model_dtype = next(iter(model.parameters())).dtype
        
        
    def chose_from_best_moves(self, board: chess.Board, best_moves):
        batch_moves = self.batch_moves(board, best_moves).cuda()

        with torch.inference_mode():
            move_scores = self.model(batch_moves).flatten()
            best_move_index = torch.argmax(move_scores).cpu().item()
            return best_moves[best_move_index]
            
    
    def batch_moves(self, board: chess.Board, moves):
        next_pos = br.next_positions(board, moves)
        board_np = br.board_to_numpy(board)
        next_board_nps = list(map(br.board_to_numpy, next_pos))
        batch_list = [np.concatenate([board_np, next_np]) for next_np in next_board_nps]
        batch_numpy = np.stack(batch_list)
        batch_tensor = torch.from_numpy(batch_numpy).to(self.model_dtype)

        return batch_tensor