# Dealing with the extended FEN format.
# <FEN position> <Next move> <White Elo> <Black Elo> <Time control>

import chess.pgn
import pandas as pd

def classify_time_control(time_control):
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



pgn_path = 'data/raw_pgn/lichess_tournament_2024.04.18_NCKOMuSh_1700-rapid.pgn'
xfen_records = parse_pgn_to_xfen(pgn_path)
xfen_df = pd.DataFrame(xfen_records)
xfen_df.to_csv('tmp/output_xfen.csv', index=False, header=False)