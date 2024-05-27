# Chessformer

The goal of this project is to create a chess engine that understand human level strength and the difficulty to find specific moves.

I hope it will be usable as a coach, to help players find the move that they could find according to their level.

I suspect that it could also be used as a cheating detection tool as it would have a better understanding of human chess as compared to a stockfish-based cheating detection tool.

Finally, I hope that I can use those models to imitate a specific player style.

I will try to achieve this by training a transformer neural network from scratch.

More specifically, as transformer models are naturally suited to sets, not for sequences, I will ask the model to predict the move of the position given a set of pieces representing the chess board and the level of the player to make a move.

I don't intend to give the model the rules of chess.

Here is an outline and more details about this personal project

I created an environment. I intend to use the following python libraries
berserk (for lichess)
chess (for handling formats)
torch (for deep learning)

And of course the basics (pandas, ...)

Here is more detail about what I want in this

Create a transformer model and train it on human chess games.
I already have several data formats that I will need to handle, so I will need to do some work to convert them to FEN, which is close to what I want to use to input my model. For the moment, I have pgn files (with multiple games) and a file with puzzles which contains the base position and then the moves to be played.
I will make files with many FEN and then use this to have train, test and val sets for both of pgn and puzzles data (to allow ablation studies and fine-grained analysis).

Then I will send these to a tokenizer. This tokenizer will have tokens for

- whose player it is to play
- Each piece (white and black)
- Each square on the board
- The Elo of the players (as classes of 100 Elo, including unknown Elo)
- The time control (classic, rapid, blitz, bullet, unknown time control)
- Whether sides can castle
- en passant of each column
- padding
- for the move to be played, from and to each square (example f_h5 and t_h5) as well as prom_n, prom_r, prom_b and prom_q if promotion is possible

The model will be autoregressive. It will be trained with the position and it will be trained to find first the square from, then the square to, and then, if needed, the promotion.

I intend to also train it on other things, maybe guessing the evaluation of the position with a regression head, or maybe to predict all the legal moves in a multi-class classification (on top of every piece, predicting squares where it is allowed to move).

For the architecture, i designed the tokenizer such that no positional embedding is needed. In fact, no positional embedding is needed for special tokens (Elo, castle etc.) but positional embeddings for pieces on the board are the square embedding for example a white Knight is on f3, the embeddings for tokens White_Knight and for token f3 will be added together.
