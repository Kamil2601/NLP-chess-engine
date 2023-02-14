import chess
import chess.pgn
import pandas as pd
import io
import numpy as np

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


def dict_from_file(filename):
    d = {}
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split(' ')
            try:
                d[line[0]] = np.array(line[1:], dtype=np.float32)
            except:
                continue
    return d



def embeddings_to_matrix_and_dict(embeddings, add_padding=True, padding_token="<PAD>"):
    """
    Converts a dictionary of pretrained word embeddings to a numpy array of embeddings and a dictionary mapping words to their indices.

    Parameters:
    - embeddings (dict): A dictionary of pretrained word embeddings, with words as keys and their embedding vectors as values.
    - add_padding (bool, optional): If True, adds an additional vector of zeros to the numpy array, representing a special token (e.g. <PAD>). Default is False.

    Returns:
    - A tuple of two elements:
        - embedding_matrix (np.ndarray): A numpy array of shape (num_words, embedding_dim), where num_words is the number of words in the dictionary and embedding_dim is the dimension of each word's embedding vector.
        - word_to_index (dict): A dictionary mapping each word in the dictionary to its index in the embedding_matrix.

    """

    words = list(embeddings.keys())
    embedding_dim = len(embeddings[words[0]])
    num_words = len(words)
    if add_padding:
        num_words += 1
        embedding_matrix = np.zeros((num_words, embedding_dim))
        embedding_matrix[1:, :] = np.array([embeddings[word] for word in words])
        word_to_index = {word: i+1 for i, word in enumerate(words)}
        word_to_index[padding_token] = 0
    else:
        embedding_matrix = np.array([embeddings[word] for word in words])
        word_to_index = {word: i for i, word in enumerate(words)}
    return embedding_matrix, word_to_index

    


