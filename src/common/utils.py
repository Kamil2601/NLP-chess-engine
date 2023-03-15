import chess
import chess.pgn
import pandas as pd
import io
import numpy as np
import sqlite3 as db

def game_to_moves(game: chess.pgn.Game):
    return [(node.parent.board().fen(), node.parent.move, node.comment) for node in game.mainline()]

def annotated_moves(game: chess.pgn.Game):
    return [(node.parent.board().fen(), node.move, node.comment, node.ply()) for node in game.mainline() if len(node.comment) > 2]

def games_to_moves(games: pd.DataFrame):
    moves_list = []
    for pgn, id in zip(games.pgn, games.id):
        game_moves = annotated_moves(chess.pgn.read_game(io.StringIO(pgn)))
        game_moves = [move + (id,) for move in game_moves]
        moves_list += game_moves

    return pd.DataFrame(moves_list, columns=["position", "move", "comment", "halfmove_number", "game_id"])


def load_sql_to_df(sql, db_filename):
    con = db.connect(db_filename)
    data_frame = pd.read_sql_query(sql, con)

    return data_frame



