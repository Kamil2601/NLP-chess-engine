import chess
import chess.pgn


good_move_nags = {1,3,7,8}
bad_move_nags = {2,4,9}

white_win_nags = {16,18,20}
black_win_nags = {17,19,21}

def get_nags(game: chess.pgn.GameNode):
    result = []

    if game.parent:
        if game.nags.intersection(good_move_nags):
            result.append((game.parent.board().fen(), game.move, 1))
        elif game.nags.intersection(bad_move_nags):
            result.append((game.parent.board().fen(), game.move, 0))

    for node in game.variations:
        result += get_nags(node)

    return result


def get_comments(game: chess.pgn.GameNode):
    result = []

    if game.parent and len(game.comment) > 0:
        if game.nags.intersection(good_move_nags):
            result.append((game.parent.board().fen(), game.move, game.comment, 1))
        elif game.nags.intersection(bad_move_nags):
            result.append((game.parent.board().fen(), game.move, game.comment, 0))
        else:
            result.append((game.parent.board().fen(), game.move, game.comment, -1))

    for node in game.variations:
        result += get_comments(node)

    return result


def load_games_from_pgn_file(pgn_file):
    """
    Load all chess games from a PGN file into a list of chess.pgn.Game objects.
    Skip games that raise an exception during loading.
    
    Args:
        pgn_file (str): the file path of the PGN file to load
        
    Returns:
        A list of chess.pgn.Game objects
    """
    games = []
    with open(pgn_file) as f:
        while True:
            try:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                games.append(game)
            except Exception as e:
                continue
    return games
