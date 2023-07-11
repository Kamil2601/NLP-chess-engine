from collections import Counter
import chess as chess
import chess.engine
import torch 
import torch.nn as nn
import playing.agents as agents
from tqdm import tqdm

engine = chess.engine.SimpleEngine.popen_uci("stockfish")

def batch_legal_moves(board: chess.Board):
    move_tensors = [br.move_to_tensor(board.fen(), move) for move in board.legal_moves]
    return torch.stack(move_tensors)

def model_best_move(board: chess.Board, model: nn.Module):
    agent = agents.ModelAgent(model)
    return agent.play(board)
    

def compare_moves(board, move1, move2):
    result1 = engine.analyse(board, chess.engine.Limit(time=0.2), root_moves=[move1])
    result2 = engine.analyse(board, chess.engine.Limit(time=0.2), root_moves=[move2])
    
    if result1["score"].relative > result2["score"].relative:
        return -1
    elif result1["score"].relative < result2["score"].relative:
        return 1
    else:
        return 0
    
def compare_model_to_random_agent(model: nn.Module, fens, random_moves_from_position = 1):
    results = []
    for fen in fens:
        board = chess.Board(fen)

        legal_moves = list(board.legal_moves)

        model_move = model_best_move(board, model)

        for _ in range(random_moves_from_position):
            random_move = random.choice(legal_moves)
            comparison = compare_moves(board, model_move, random_move)
            results.append(comparison)

    stats = Counter(results)

    return {"model_better": stats[-1], "random_better": stats[1], "draw": stats[0]}


def play_game(white: agents.Agent, black: agents.Agent, n_moves = 40, verbose = False):
    board = chess.Board()

    if verbose:
        display(board)

    for i in range(n_moves):
        for agent in [white, black]:
            if board.is_game_over():
                break

            move = agent.play(board)

            board.push(move)

            if verbose:
                display(board)

    return board

def simple_result(board, time = 1):
    score = engine.analyse(board, chess.engine.Limit(time=time))['score'].white()

    if score <= chess.engine.Cp(-100):
        return -1
    elif score < chess.engine.Cp(100):
        return 0
    return 1


def test_agent(model_agent, n_games = 50, n_moves = 40, time = 1):
    random_agent = agents.RandomAgent()
    results = []

    white = model_agent
    black = random_agent
    model_white = 1


    for game in tqdm(range(n_games)):
        final_board = play_game(white, black, n_moves=40)
        score = model_white * simple_result(final_board, time=time)
        results.append(score)

        white, black = black, white
        model_white = -model_white


    stats = Counter(results)

    return {"model_better": stats[1], "random_better": stats[-1], "draw": stats[0]}