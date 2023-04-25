import chess
import chess.pgn
import pandas as pd
import io
import numpy as np
import sqlite3 as db
import matplotlib.pyplot as plt

def game_to_moves(game: chess.pgn.Game):
    return [(node.parent.board().fen(), node.parent.move, node.comment) for node in game.mainline()]

def annotated_moves(game: chess.pgn.Game):
    return [(str(node.parent.board().fen()), str(node.move), str(node.comment), node.ply()) for node in game.mainline() if len(node.comment) > 2]

def games_to_moves_gameknot(games: pd.DataFrame):
    moves_list = []
    for pgn, id in zip(games.pgn, games.id):
        game_moves = annotated_moves(chess.pgn.read_game(io.StringIO(pgn)))
        game_moves = [move + (id,) for move in game_moves]
        moves_list += game_moves

    return pd.DataFrame(moves_list, columns=["position", "move", "comment", "halfmove_number", "game_id"])

def games_to_moves(games: list):
    moves_list = []
    for pgn in games:
        game_moves = annotated_moves(chess.pgn.read_game(io.StringIO(pgn)))
        moves_list += game_moves

    return pd.DataFrame(moves_list, columns=["position", "move", "comment", "halfmove_number"])


def load_sql_to_df(sql, db_filename):
    con = db.connect(db_filename)
    data_frame = pd.read_sql_query(sql, con)
    con.close()

    return data_frame

def save_to_sql(df, db_filename, table_name, if_exists='fail'):
    con = db.connect(db_filename)
    df.to_sql(table_name, con, if_exists=if_exists)
    con.close()
    return


def plot_history(history: dict):
    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot(1, 2, 1)
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.legend(['Train loss', 'Validation loss'], fontsize=10)
    ax.set_xlabel('Epochs', size=15)
    ax = fig.add_subplot(1, 2, 2)
    plt.plot(history['train_accuracy'])
    plt.plot(history['val_accuracy'])
    plt.legend(['Train acc.', 'Validation acc.'], fontsize=10)
    ax.set_xlabel('Epochs', size=15)
