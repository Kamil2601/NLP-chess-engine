import torch
import torch.nn as nn

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
    def __init__(self, model: nn) -> None:
        super().__init__()
        self.model = model
        self.model.cpu()

    