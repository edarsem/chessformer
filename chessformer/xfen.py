# Dealing with the extended FEN format.
# <FEN position> <Next move> <White Elo> <Black Elo> <Time control>

import chess.pgn

def classify_time_control(time_control):
    """
    Classify the time control of a chess game.

    Args:
        time_control (str): The time control string from the PGN headers.

    Returns:
        str: The classified time control (bullet, blitz, rapid, classical, unknown).
    """
    if time_control == "-" or not time_control:
        return "Unknown"
    elif '+' in time_control:
        base, increment = map(int, time_control.split('+'))
        total_time = base + 60 * increment
    else:
        total_time = int(time_control)
    
    if total_time < 180:
        return "bullet"
    elif total_time < 600:
        return "blitz"
    elif total_time < 1500:
        return "rapid"
    else:
        return "classical"

def parse_pgn_to_xfen(pgn_file):
    """
    Parse PGN file to extract XFEN data.

    Args:
        pgn_file (str): Path to the PGN file.

    Returns:
        list: A list of dictionaries containing XFEN data.
    """
    pgn = open(pgn_file, 'r')
    games = []

    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break
        headers = game.headers
        board = chess.Board()
        time_control = classify_time_control(headers.get('TimeControl', '-'))
        white_elo = headers.get('WhiteElo', 'Unknown')
        black_elo = headers.get('BlackElo', 'Unknown')

        for move in game.mainline_moves():
            xfen_data = {
                'XFEN': board.fen(),
                'Next Move': move.uci(),
                'White Elo': white_elo,
                'Black Elo': black_elo,
                'Time Control': time_control
            }
            games.append(xfen_data)
            board.push(move)
    
    return games