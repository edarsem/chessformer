import chess
import chess.pgn
import io

import download_games

num_to_letter = {
    1: 'a',
    2: 'b',
    3: 'c',
    4: 'd',
    5: 'e',
    6: 'f',
    7: 'g',
    8: 'h'
    }

def fen_to_pretokens(fen):
    pretokens = []
    fen_parts = fen.split()
    # Extract color to move
    color_to_move = fen_parts[1]
    pretokens.append(color_to_move)
    # Extract castling rights
    castling_rights = fen_parts[2]
    for right in castling_rights:
        if right in 'KQkq':
            pretokens.append(right)
    # For each piece, extract the square it is on
    board_rows = fen_parts[0].split("/")
    for index_row, row in enumerate(board_rows):
        real_index_col = 0
        for col in row:
            if col.isdigit():
                real_index_col += int(col)
            else:
                real_index_col += 1
                pretokens.append(f"{col}{num_to_letter[real_index_col]}{8-index_row}")
    return pretokens

def pgn_to_fen(game_data):
    pre_tokenized_list = []
    pgn = game_data.split("\n\n")
    for game_str in pgn:
        game_string_io = io.StringIO(game_str)
        game = chess.pgn.read_game(game_string_io)
        board = game.board()
        for move in game.mainline_moves():
                board.push(move)
                fen = board.fen()
                pre_tokenized = fen_to_pretokens(fen)
                pre_tokenized_list.append(pre_tokenized)

# pgn_to_fen(download_games.download_games())

# fen_to_pretokens('8/5ppp/pk2p3/1nn1P2P/3rP1P1/6K1/8/1R6 w - - 2 43')