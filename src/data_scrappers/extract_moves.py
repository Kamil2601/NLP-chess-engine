import sys
sys.path.append("..")

import os
import chess.pgn
import chess
import pandas as pd
import pathlib
from tqdm.auto import tqdm


from utils.utils import games_to_moves, save_to_sql

def get_pgn_files(path, recursive=True):
    if isinstance(path, str):
        path = pathlib.Path(path)

    if not path.exists():
        return []
    if path.is_file():
        if path.suffix == ".pgn":
            return [path]
        else:
            return []

    if recursive:
        return list(path.glob("**/*.pgn"))
    else:
        return list(path.glob("*.pgn"))

def extract_games(path):
    pgn_files = get_pgn_files(path)
    games = []
    for pgn_file in tqdm(pgn_files, desc="Extracting games"):
        with open(pgn_file) as f:
            while True:
                try:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    games.append(game)
                except:
                    pass

    return games



good_move_nags = {1,3,7,8}
bad_move_nags = {2,4,6,9}

white_win_nags = {16,18,20}
black_win_nags = {17,19,21}

def sentiment_from_nags(nags):
    if nags.intersection(good_move_nags):
        return 1
    elif nags.intersection(bad_move_nags):
        return 0
    else:
        return -1

def extract_info(node):
    info = {
        'fen': node.parent.board().fen(),
        'move': node.move.uci(),
        'comment': node.comment,
        'nags': list(node.nags),
        'sentiment': sentiment_from_nags(node.nags)
    }
    return info

def filter_by_comment(node):
    return node.comment and not node.comment.isspace() and len(node.comment) > 1

def has_comment_or_nag(node):
    return filter_by_comment(node) or node.nags

def extract_info_for_all_nodes(game, extract_func = extract_info, filter_func = has_comment_or_nag):
    all_info = []
    
    def traverse(node):
        if node.parent and filter_func(node):
            all_info.append(extract_func(node))
        if node.variations:
            for var in node.variations:
                traverse(var)
    
    traverse(game)
    return all_info


def extract_games_info(games, extract_func = extract_info, filter_func = has_comment_or_nag):
    games_info = []
    for game in tqdm(games, desc="Extracting moves info from games"):
        games_info += extract_info_for_all_nodes(game, extract_func=extract_func, filter_func=filter_func)
    return games_info

def extract_games_info_from_pgn_files(path, extract_func = extract_info, filter_func = has_comment_or_nag):
    games = extract_games(path)
    return extract_games_info(games, extract_func=extract_func, filter_func=filter_func)