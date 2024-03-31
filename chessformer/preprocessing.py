import chess
import chess.pgn
import io

import download_games

num_to_letter = dict(zip(range(1,9), 'abcdefgh'))

# In training mode, returns a list of all steps to generate (the from move, the to move, and the promotion) with the move to generate being in last position
def fen_to_pretokens(fen, move, training_mode=False):
    assert len(move) in (4, 5) and move[0] in 'abcdefgh' and move[1] in '12345678' and move[2] in 'abcdefgh' and move[3] in '12345678', f"{move} is an invalid move"
    move_from = f"f_{move[:2]}"
    move_to = f"t_{move[2:4]}"
    promotion = move[4] if len(move) == 5 else None
    pretokens = []
    all_steps = []
    fen_parts = fen.split()
    # Extract color to move
    color_to_move = fen_parts[1]
    pretokens.append(color_to_move)
    # Extract castling rights
    castling_rights = fen_parts[2]
    for right in castling_rights:
        if right in 'KQkq':
            pretokens.append(right)
    # Extract en passant square
    en_passant = fen_parts[3]
    if en_passant != '-':
        pretokens.append(f"ep_{en_passant[0]}")
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
    pretokens.append(move_from)
    all_steps.append(pretokens.copy())
    pretokens.append(move_to)
    all_steps.append(pretokens.copy())
    if promotion:
        pretokens.append(f'prom_{promotion}')
        all_steps.append(pretokens.copy())
    if training_mode:
        return all_steps
    return pretokens

def pgn_to_fen(game_data):
    fen_list = []
    pgn = game_data.split("\n\n")
    for game_str in pgn:
        game_string_io = io.StringIO(game_str)
        game = chess.pgn.read_game(game_string_io)
        board = game.board()
        for move in game.mainline_moves():
            fen = board.fen()
            if move.uci() != '0000':
                fen_list.append((fen, move.uci()))
            board.push(move)
    return fen_list

# Convert a list of fens to a list of pretokens. In training mode, does this for every step of each move
def fen_list_to_pretokens_list(fen_list, training_mode=False):
    pretokens_list = []
    for fen in fen_list:
        pretokens = fen_to_pretokens(fen[0], fen[1], training_mode)
        if training_mode:
            pretokens_list += pretokens
        else:
            pretokens_list.append(pretokens)
    return pretokens_list