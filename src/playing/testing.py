from collections import Counter
from pstats import Stats
import chess as chess
import chess.engine
import torch
import torch.nn as nn
import playing.agents as agents
from tqdm import tqdm
from IPython.display import display
import board_representation as br
import random
from .utils import material
import pandas as pd

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


def compare_model_to_random_agent(model: nn.Module, fens, random_moves_from_position=1):
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


def play_game(white: agents.Agent, black: agents.Agent, n_moves=40, verbose=False):
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
                # print(board.fen())
                display(board)

    return board


def simple_result(board, time=1):
    score = engine.analyse(board, chess.engine.Limit(time=time))["score"].white()

    if score <= chess.engine.Cp(-100):
        return -1
    elif score < chess.engine.Cp(100):
        return 0
    return 1


def test_agent(agent, opponent=None, n_games=50, n_moves=40, time=1):
    if not opponent:
        opponent = agents.RandomAgent()

    white_scores = []
    for game in tqdm(range(n_games // 2)):
        final_board = play_game(agent, opponent, n_moves=40)
        score = simple_result(final_board, time=time)
        white_scores.append(score)

    white_stats = Counter(white_scores)
    print(white_stats)

    black_scores = []
    for game in tqdm(range(n_games // 2)):
        final_board = play_game(opponent, agent, n_moves=40)
        score = simple_result(final_board, time=time)
        black_scores.append(score)

    black_stats = Counter(black_scores)
    print(black_stats)

    return {"white": white_stats, "black": black_stats}

    # return {"agent_win": stats[1], "opponent_win": stats[-1], "draw": stats[0]}


def play_game_with_material_count(
    white: agents.Agent, black: agents.Agent, n_moves=40, verbose=False
):
    board = chess.Board()

    if verbose:
        display(board)

    white_materials = []
    black_materials = []

    white_material, black_material = material(board)
    white_materials.append(white_material)
    black_materials.append(black_material)

    for i in range(n_moves):
        for agent in [white, black]:
            if board.is_game_over():
                break

            move = agent.play(board)

            board.push(move)

            if verbose:
                display(board)

        white_material, black_material = material(board)
        white_materials.append(white_material)
        black_materials.append(black_material)

    return board, white_materials, black_materials


def test_agent_material_count(agent, opponent=None, n_games=50, n_moves=40, time=1):
    if not opponent:
        opponent = agents.RandomAgent()
    results = []

    agent_materials = []
    opponent_materials = []

    white_scores = []
    for game in tqdm(range(n_games // 2)):
        final_board, white_materials, black_materials = play_game_with_material_count(
            agent, opponent, n_moves=40
        )
        score = simple_result(final_board, time=time)
        white_scores.append(score)

        agent_materials.append(white_materials)
        opponent_materials.append(black_materials)

    white_stats = Counter(white_scores)
    print(white_stats)

    black_scores = []
    for game in tqdm(range(n_games // 2)):
        final_board, white_materials, black_materials = play_game_with_material_count(
            opponent, agent, n_moves=40
        )
        score = simple_result(final_board, time=time)
        black_scores.append(score)

        agent_materials.append(black_materials)
        opponent_materials.append(white_materials)

    black_stats = Counter(black_scores)
    print(black_stats)

    agent_materials = list(pd.DataFrame(agent_materials).mean(axis=0, skipna=True))
    opponent_materials = list(pd.DataFrame(opponent_materials).mean(axis=0, skipna=True))

    return {"white": white_stats, "black": black_stats}, agent_materials, opponent_materials



def play_against_agent(agent, color=chess.WHITE):
    board = chess.Board()
    display(board)
    while not board.is_game_over():
        if board.turn != color:
            move = agent.play(board)
            board.push(move)
        else:
            while True:
                try:
                    move = input("Enter move (UCI): ")
                    if move == "q":
                        return
                    move = chess.Move.from_uci(move)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print("Illegal move")
                except:
                    print("Illegal move")
        display(board)
    print("Game over")
    print(board.result())