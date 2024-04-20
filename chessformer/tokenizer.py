import json

all_tokens_file = 'tmp/token_to_id.json'

def create_token_mapping(file):
    # Define the range of Elos and other categories
    elo_ranges = list(range(800, 3401, 100)) + ['weak', 'strong', 'unknown']
    time_controls = ['bullet', 'blitz', 'rapid', 'classical', 'unknown_time']

    # Create a dictionary for piece placement from FEN
    pieces = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']
    squares = [f"{chr(file)}{rank}" for rank in range(1, 9) for file in range(97, 105)]
    moves_from = [f"f_{square}" for square in squares]
    moves_to = [f"t_{square}" for square in squares]
    promotion = ['promote_n', 'promote_b', 'promote_r', 'promote_q']
    castling = ['w_k', 'w_q', 'b_k', 'b_q']
    en_passant = ['ep_' + col for col in 'abcdefgh']

    # Combine all tokens
    tokens = ['play_w', 'play_b'] + time_controls + [f"elo_{elo}" for elo in elo_ranges] + castling + en_passant + pieces + squares + moves_from + moves_to + promotion

    # Create a token ID dictionary
    token_to_id = {token: idx for idx, token in enumerate(tokens)}

    # Save to file
    with open(file, 'w') as f:
        json.dump(token_to_id, f)

def tokenize_xfen(xfen_row, token_to_id, inferrence=False):
    """
    Tokenize an extended FEN row.
    The 2 first elements are the number of tokens dedicated to the next move and the number of pieces on the board.
    Maximum 42 tokens as input for a position (1 for the player to move, 1 for the Elo, 1 for the player id if any, 1 for the time control, 4 for castling rights, 1 for en passant, 32 for the board, and 2 to 3 for the move itself but it can't be 3 if there are 32 pieces left and minus 1 because the token in prediction is not as input).
    Inferrence mode when the next move is not available."""
    fen, next_move, white_elo, black_elo, time_control = xfen_row.split(',')

    # Parse the FEN part
    board, turn, castling, en_passant, _, _ = fen.split(' ')

    # Tokenize the board setup
    pieces_tokens = []
    squares_tokens = []
    for rank_index, row in enumerate(board.split('/')):
        file_index = 0
        for char in row:
            if char.isdigit():
                file_index += int(char)
            elif char.isalpha():
                square_index = f"{chr(97 + file_index)}{8 - rank_index}"
                pieces_tokens.append(token_to_id[char])
                squares_tokens.append(token_to_id[square_index])
                file_index += 1

    to_move = [token_to_id[f'play_{turn}']]
    castling_tokens = [
            token_to_id['w_k'] if 'K' in castling else None,
            token_to_id['w_q'] if 'Q' in castling else None,
            token_to_id['b_k'] if 'k' in castling else None,
            token_to_id['b_q'] if 'q' in castling else None
        ]
    castling_tokens = [token for token in castling_tokens if token is not None]
    en_passant_tokens = [token_to_id['ep_' + en_passant[0]]] if en_passant != '-' else []
    elo_token = [token_to_id[get_elo_token(white_elo)] if turn == 'w' else token_to_id[get_elo_token(black_elo)]]    
    time_control_token = [token_to_id[time_control]]
    
    meta_tokens = to_move + elo_token + time_control_token + castling_tokens + en_passant_tokens

    if inferrence:
        return meta_tokens, pieces_tokens, squares_tokens, None

    from_square, to_square = next_move[:2], next_move[2:4]
    move_tokens = [token_to_id[f'f_{from_square}'], token_to_id[f't_{to_square}']]
    
    if len(next_move) > 4:
        promotion_piece = 'promote_' + next_move[-1].lower()
        move_tokens.append(token_to_id[promotion_piece])
    
    res = []
    for id_next_move in range(len(move_tokens)):
        res.append([meta_tokens + move_tokens[:id_next_move], pieces_tokens, squares_tokens, move_tokens[id_next_move]])

    return res

def get_elo_token(elo):
    if elo.lower() == 'unknown':
        return "elo_unknown"
    elo = int(elo)
    if elo < 800:
        return "elo_weak"
    elif elo >= 3400:
        return "elo_strong"
    else:
        return f"elo_{elo // 100 * 100}"

def detokenize_xfen(tokens, id_to_token):
    return [id_to_token[token] for token in tokens]

def tokenize_file(input_file, output_file, token_to_id):
    with open(input_file, 'r') as f:
        data = f.readlines()

    tokenized_data = [','.join(map(str, tokenize_xfen(row.strip(), token_to_id))) for row in data]

    with open(output_file, 'w') as f:
        f.write('\n'.join(tokenized_data))

def load_token_mapping(file=all_tokens_file):
    with open(file, 'r') as f:
        token_to_id = json.load(f)
    return token_to_id

def tokenize_batch(batch, token_to_id, inferrence=False):
    if inferrence:
        return [tokenize_xfen(xfen_row, token_to_id, inferrence) for xfen_row in batch]
    # else, we need to loop over the result which is a list instead of one element
    return [elt for xfen_row in batch for elt in tokenize_xfen(xfen_row, token_to_id, inferrence)]

def process_xfen_file_in_batches(input_file, output_file, token_to_id, batch_size=1000, inferrence=False):
    output_handle = open(output_file, 'w')

    with open(input_file, 'r') as file:
        batch = []
        for line in file:
            batch.append(line.strip())
            if len(batch) >= batch_size:
                tokenized_batch = tokenize_batch(batch, token_to_id, inferrence=inferrence)
                for tokenized_entry in tokenized_batch:
                    output_handle.write(json.dumps(tokenized_entry) + '\n')
                batch = []

        if batch:
            tokenized_batch = tokenize_batch(batch, token_to_id)
            for tokenized_entry in tokenized_batch:
                output_handle.write(json.dumps(tokenized_entry) + '\n')

    output_handle.close()

# # Tokenize a file
# input_file = 'tmp/output_xfen.csv'
# output_file = 'tmp/tokenized_xfen.txt'
# token_to_id = load_token_mapping(all_tokens_file)
# process_xfen_file_in_batches(input_file, output_file, token_to_id, batch_size=1000)