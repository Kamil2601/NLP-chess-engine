from bs4 import BeautifulSoup
import requests
import chess
import chess.pgn
import requests
import json
import numpy as np
import urllib.parse as up
import re
from tqdm import tqdm
import sqlite3
import os
import time

def download_gameknot_game(id):
    url = f"https://gameknot.com/annotate.pl?id={id}"
    results = requests.get(url)
    lines = results.text.split('\n')
    game_movelist = None
    game_notes = None

    game = chess.pgn.Game()
    node = game
    game.headers.clear()

    result_dict = ["", "1-0", "1/2-1/2", "0-1"]

    def fix_move(move):
        if move[-1] == "-":
            return move[:-1]
        return move

    for line in lines:
        if line.startswith('game_movelist'):
            game_str = eval(line[16:-1])
            game_movelist = [game_str[i: i+5] for i in range(0, len(game_str), 5)]
            game_movelist = [fix_move(move) for move in game_movelist]
        elif line.startswith('game_notes'):
            game_notes = list(map(up.unquote, eval(line[13:-1])))
        elif line.startswith('game_result'):
            game.headers["Result"] = result_dict[int(line[-2])]
        elif line.startswith('game_title'):
            game.headers["Event"] = up.unquote(line[25:-3])
        elif line.startswith('game_player_w'):
            game.headers["White"] = up.unquote(line[28:-3])
        elif line.startswith('game_player_b'):
            game.headers["Black"] = up.unquote(line[28:-3])
        elif line.startswith('game_rating_w'):
            game.headers["WhiteElo"] = up.unquote(line[28:-3])
        elif line.startswith('game_rating_b'):
            game.headers["BlackElo"] = up.unquote(line[28:-3])
        elif line.startswith('game_started'):
            game.headers["Date"] = up.unquote(line[27:-3])
        


    node.comment = game_notes[0]


    for move, comment in  zip(game_movelist[:-1], game_notes[1:]):
        node = node.add_variation(chess.Move.from_uci(move))
        node.comment = comment

    return game

def load_ids_from_file(file_name = "data_scrappers/annotated_game_ids.txt"):
    f = open(file_name)
    return list(map(int, f.readlines()))


def download_all_games(table_name = "gameknot_games_2"):
    db = sqlite3.connect('./chess.db')

    cur = db.cursor()

    cur.execute(f'''DROP TABLE IF EXISTS {table_name}''')

    cur.execute(f'''CREATE TABLE {table_name}(
        id integer PRIMARY KEY,
        pgn text NOT NULL);''')

    all_ids = load_ids_from_file()
    failed_ids = []
    game_list = []
    errors_in_row = 0


    for i, id in enumerate(tqdm(all_ids)):
        if i % 50 == 0:
            time.sleep(3)
        try:
            game = download_gameknot_game(id)
            if game != None:
                game_list.append((id, str(game)))
                print(game, file=open(f"./data/gameknot/annotated_games_2/{id}.pgn", "w"), end="\n\n")
                errors_in_row = 0
        except:
            print(id)
            errors_in_row += 1
            failed_ids.append(id)

        if errors_in_row == 10:
            print(f"ERROR after {i-10} iterations")

    print(game, file=open(f"failed_ids.txt", "w"), end="\n\n")

    cur.executemany(f"INSERT INTO {table_name}(id, pgn) VALUES(?,?)", game_list)  
    
    db.commit()
    db.close()


def copy_games_files_to_database(table_name = "gameknot_games"):
    db = sqlite3.connect('./chess.db')

    cur = db.cursor()

    cur.execute(f'''DROP TABLE IF EXISTS {table_name}''')

    cur.execute(f'''CREATE TABLE {table_name}(
        id integer PRIMARY KEY,
        pgn text NOT NULL);''')

    path = "data/gameknot/annotated_games"
    dirs = os.listdir(path)
    

    game_list = []

    for game_file in dirs:
        pgn = open(f"data/gameknot/annotated_games/{game_file}").read()
        # game = chess.pgn.read_game(pgn)
        id = int(game_file[:-4])
        game_list.append((id,pgn))

    cur.executemany(f"INSERT INTO {table_name}(id, pgn) VALUES(?,?)", game_list)  
    
    db.commit()
    db.close()

def main():
    download_all_games()
    # copy_games_files_to_database()

if __name__ == "__main__":
    main()
