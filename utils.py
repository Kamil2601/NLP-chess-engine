import chess
import pandas as pd
import itertools

def game_to_moves(game: chess.pgn.Game):
    return [(node.parent.board().fen(), node.parent.move, node.comment) for node in game.mainline()]

def annotated_moves(game: chess.pgn.Game):
    return [(node.parent.board().fen(), node.parent.move, node.comment) for node in game.mainline() if len(node.comment) > 2]

# def games_to_moves(games: pd.DataFrame):
    

