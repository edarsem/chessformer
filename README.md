# Chessformer

The goal of this project is to create a chess engine that understand human level strength and the difficulty to find specific moves. I hope it will be usable as a coach, to help players find the move that they could find according to their level. I suspect that it could also be used as a cheating detection tool as it would have a better understanding of human chess as compared to a stockfish-based cheating detection tool.

I will try to achieve this by trainnig a transformer neural network from scratch.

More specifically, as transformer models are naturally suited to sets, not for sequences, I will ask the model to predict the move of the position given a set of pieces representing the chess board and the level of the player to make a move.

I don't intend to give the model the rules of chess.