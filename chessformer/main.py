from time import time

import model
import tokenizer
import preprocessing
import download_games

# config = model.GPTConfig()
# chessformer = model.GPT(config)


# print(model)

tokenizer = tokenizer.Tokenizer('all_tokens.txt')

user_name = 'nihalsarin2004'

games = download_games.download_games(user_name, days_back=50)

# print(games)

fen_list = preprocessing.pgn_to_fen(games)

time0 = time()
verif = 0
for fen in fen_list:
    pretokens = preprocessing.fen_to_pretokens(fen[0], fen[1])
    tokens = tokenizer.encode(pretokens)
    pretokens_res = tokenizer.decode(tokens)
    if pretokens_res == pretokens:
        verif += 1
    else:
        print(fen)
        print(pretokens)
        print(pretokens_res)
        print('\n')

print((time() - time0)/len(fen_list))

print(len(fen_list))

print(verif)