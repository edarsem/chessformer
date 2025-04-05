import torch
import chess
import chess.svg
import random
import numpy as np
from IPython.display import SVG, display

from model import ChessModel
from tokenizer import tokenize_xfen, load_token_mapping
from config import DEVICE, MODEL_PARAMS, TOKEN_TO_ID_PATH

class ChessAI:
    """
    AI for playing chess using the ChessFormer transformer model.
    Includes sampling strategies for more creative play.
    """
    def __init__(self, model=None, token_to_id=None, id_to_token=None, elo="2000", time_control="blitz"):
        """
        Initialize the Chess AI with model and tokenization.
        
        Args:
            model: The ChessFormer model
            token_to_id: Dictionary mapping tokens to IDs
            id_to_token: Dictionary mapping IDs to tokens
            elo: ELO rating for the AI
            time_control: Time control setting (bullet, blitz, rapid, classical)
        """
        # Load model if not provided
        if model is None:
            self.model = ChessModel(
                num_tokens=MODEL_PARAMS['num_tokens'],
                num_classes=MODEL_PARAMS['num_classes'],
                d_model=MODEL_PARAMS['d_model'],
                nhead=MODEL_PARAMS['nhead'],
                num_layers=MODEL_PARAMS['num_layers'],
                dropout=MODEL_PARAMS['dropout']
            )
            try:
                self.model.load_state_dict(torch.load('model_weights.pt'))
                print("Model weights loaded successfully!")
            except FileNotFoundError:
                print("No model weights found. Using untrained model.")
            
            self.model.to(DEVICE)
            self.model.eval()
        else:
            self.model = model
        
        # Load token mappings if not provided
        if token_to_id is None or id_to_token is None:
            try:
                self.token_to_id = load_token_mapping(TOKEN_TO_ID_PATH)
                self.id_to_token = {v: k for k, v in self.token_to_id.items()}
                print(f"Token mapping loaded successfully from {TOKEN_TO_ID_PATH}")
            except FileNotFoundError:
                print(f"Error: Token mapping file not found at {TOKEN_TO_ID_PATH}")
                raise
        else:
            self.token_to_id = token_to_id
            self.id_to_token = id_to_token
            
        self.elo = elo
        self.time_control = time_control
        
    def predict_step_with_sampling(self, fen, move_history=None, temperature=1.0):
        """
        Predict a single step of a move (from/to/promotion) using sampling.
        
        Args:
            fen (str): The FEN representation of the board
            move_history (list): List of previous move token predictions
            temperature (float): Temperature for sampling (higher = more random)
            
        Returns:
            str: The predicted token
        """
        # Create an XFEN string (FEN + additional metadata)
        xfen = f"{fen},e2e4,{self.elo},{self.elo},{self.time_control}"  # We need a dummy move
        
        try:
            # Tokenize the XFEN
            tokens_data = tokenize_xfen(xfen, self.token_to_id, inferrence=True)
            if isinstance(tokens_data, tuple):
                meta_tokens, pieces_tokens, squares_tokens, _ = tokens_data
            else:
                return "Error in tokenization: unexpected format"
            
            # Add any move history tokens
            if move_history:
                meta_tokens = meta_tokens + [self.token_to_id[token] for token in move_history if token in self.token_to_id]
                
            # Combine tokens and convert to tensors
            sequence_tokens = torch.tensor(meta_tokens + pieces_tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)
            squares_tokens_tensor = torch.tensor([-1] * len(meta_tokens) + squares_tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)
            
            # Get model prediction with sampling
            with torch.no_grad():
                logits = self.model(sequence_tokens, squares_tokens_tensor)
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Convert logits to probabilities
                probs = torch.softmax(logits, dim=-1)
                
                # Sample from the distribution
                prediction_idx = torch.multinomial(probs, num_samples=1).item()
                prediction = prediction_idx + self.model.first_class_token
                predicted_token = self.id_to_token[prediction]
                
            return predicted_token
            
        except Exception as e:
            return f"Error in prediction: {str(e)}"
            
    def predict_full_move_with_sampling(self, fen, temperature=1.0, max_attempts=5, verbose=False):
        """
        Predict a complete move by running the model multiple times with sampling.
        Will attempt to find a legal move by sampling multiple times if needed.
        
        Args:
            fen (str): The FEN representation of the board
            temperature (float): Temperature for sampling (higher = more random)
            max_attempts (int): Maximum number of sampling attempts to find a legal move
            verbose (bool): Whether to print detailed prediction info
            
        Returns:
            str: UCI move string or None if no legal move could be found
        """
        board = chess.Board(fen)
        legal_moves = [move.uci() for move in board.legal_moves]
        
        if not legal_moves:
            if verbose:
                print("No legal moves available (checkmate or stalemate)")
            return None
        
        # Just return a random legal move if there's only one option
        if len(legal_moves) == 1:
            if verbose:
                print(f"Only one legal move: {legal_moves[0]}")
            return legal_moves[0]
            
        # Try multiple times to generate a legal move
        for attempt in range(max_attempts):
            if verbose:
                print(f"Attempt {attempt+1}/{max_attempts}")
                
            # Step 1: Predict starting square
            from_square_token = self.predict_step_with_sampling(fen, None, temperature)
            
            if not isinstance(from_square_token, str) or not from_square_token.startswith('f_'):
                if verbose:
                    print(f"Invalid from square prediction: {from_square_token}")
                continue
            
            from_square = from_square_token[2:]  # Remove 'f_' prefix
            if verbose:
                print(f"Predicted from square: {from_square}")
            
            # Step 2: Predict destination square
            to_square_token = self.predict_step_with_sampling(fen, [from_square_token], temperature)
            
            if not isinstance(to_square_token, str) or not to_square_token.startswith('t_'):
                if verbose:
                    print(f"Invalid to square prediction: {to_square_token}")
                continue
            
            to_square = to_square_token[2:]  # Remove 't_' prefix
            if verbose:
                print(f"Predicted to square: {to_square}")
            
            # Form the basic move
            uci_move = f"{from_square}{to_square}"
            
            # Step 3: Check if promotion is needed
            # A move needs promotion if a pawn moves to the 8th rank or 1st rank
            piece = board.piece_at(chess.parse_square(from_square))
            to_rank = int(to_square[1])
            
            # Check if it's a pawn moving to the last rank
            if piece and piece.piece_type == chess.PAWN and (to_rank == 8 or to_rank == 1):
                promotion_token = self.predict_step_with_sampling(
                    fen, [from_square_token, to_square_token], temperature
                )
                
                if not isinstance(promotion_token, str) or not promotion_token.startswith('promote_'):
                    if verbose:
                        print(f"Invalid promotion prediction: {promotion_token}")
                    # Default to queen promotion
                    uci_move += "q"
                else:
                    promotion_piece = promotion_token.split('_')[1]
                    uci_move += promotion_piece
                    if verbose:
                        print(f"Predicted promotion: {promotion_piece}")
            
            # Verify the move is legal
            try:
                move = chess.Move.from_uci(uci_move)
                if move in board.legal_moves:
                    if verbose:
                        print(f"Legal move found: {uci_move}")
                    return uci_move
                else:
                    if verbose:
                        print(f"Illegal move: {uci_move}")
            except ValueError:
                if verbose:
                    print(f"Invalid UCI format: {uci_move}")
        
        # If we've exhausted all attempts and haven't found a legal move,
        # fall back to a random legal move
        random_move = random.choice(legal_moves)
        if verbose:
            print(f"No legal move found after {max_attempts} attempts. Using random move: {random_move}")
        return random_move

class ChessGame:
    """
    A chess game that can be played against the AI or as AI vs AI.
    """
    def __init__(self, fen=None, white_ai=None, black_ai=None):
        """
        Initialize a chess game.
        
        Args:
            fen (str): Starting position in FEN format
            white_ai (ChessAI): AI for white player (None means human)
            black_ai (ChessAI): AI for black player (None means human)
        """
        self.board = chess.Board(fen) if fen else chess.Board()
        self.move_history = []
        self.white_ai = white_ai
        self.black_ai = black_ai
        
    def display(self):
        """Display the current board position."""
        display(SVG(chess.svg.board(board=self.board, size=400)))
        print(f"FEN: {self.board.fen()}")
        if self.board.is_game_over():
            print(f"Game over: {self.board.result()}")
            
    def make_move(self, move_uci):
        """Make a move on the board."""
        try:
            move = chess.Move.from_uci(move_uci)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.move_history.append(move_uci)
                return True
            else:
                print(f"Illegal move: {move_uci}")
                return False
        except ValueError:
            print(f"Invalid move format: {move_uci}. Use UCI format (e.g., 'e2e4').")
            return False
            
    def get_ai_move(self, ai, temperature=0.8, verbose=False):
        """Get a move from the AI player."""
        if not ai:
            print("No AI player configured.")
            return None
            
        print("AI is thinking...")
        
        # Get move with sampling
        uci_move = ai.predict_full_move_with_sampling(
            self.board.fen(), temperature=temperature, verbose=verbose
        )
        
        if uci_move:
            print(f"AI plays: {uci_move}")
            return uci_move
        else:
            print("AI couldn't find a move.")
            return None
            
    def play_turn(self, temperature=0.8, verbose=False):
        """Play a single turn in the game."""
        if self.board.is_game_over():
            return False
            
        # Display current board state
        self.display()
        
        # Determine which AI to use based on whose turn it is
        current_ai = self.white_ai if self.board.turn == chess.WHITE else self.black_ai
        
        if current_ai:
            # AI's turn
            print(f"{'White' if self.board.turn == chess.WHITE else 'Black'}'s turn (AI)")
            move = self.get_ai_move(current_ai, temperature, verbose)
            if move:
                self.make_move(move)
                return True
            else:
                print("AI failed to make a move.")
                return False
        else:
            # Human's turn
            print(f"{'White' if self.board.turn == chess.WHITE else 'Black'}'s turn (Human)")
            while True:
                try:
                    move = input("Enter your move (UCI format, e.g. 'e2e4') or 'quit': ")
                    if move.lower() in ['quit', 'exit']:
                        return False
                    if self.make_move(move):
                        return True
                except Exception as e:
                    print(f"Error: {str(e)}")
    
    def play_game(self, max_moves=100, temperature=0.8, verbose=False):
        """Play a full game until completion or max moves reached."""
        move_count = 0
        
        while not self.board.is_game_over() and move_count < max_moves:
            if not self.play_turn(temperature, verbose):
                break
            move_count += 1
        
        # Final display of the board
        self.display()
        
        # Show game outcome
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            print(f"{winner} wins by checkmate!")
        elif self.board.is_stalemate():
            print("Game drawn by stalemate!")
        elif self.board.is_insufficient_material():
            print("Game drawn due to insufficient material!")
        elif self.board.is_fifty_moves():
            print("Game drawn by fifty-move rule!")
        elif self.board.is_repetition():
            print("Game drawn by repetition!")
        elif move_count >= max_moves:
            print(f"Game stopped after {max_moves} moves.")
        else:
            print("Game ended.")
            
        return {
            'result': self.board.result(),
            'moves': self.move_history,
            'final_fen': self.board.fen(),
            'move_count': move_count
        }

    def ai_vs_ai_game(self, max_moves=100, white_temp=0.8, black_temp=0.8, verbose=False):
        """Play a game between two AI players."""
        if not self.white_ai or not self.black_ai:
            print("Both AIs must be configured for AI vs AI play.")
            return
            
        move_count = 0
        
        while not self.board.is_game_over() and move_count < max_moves:
            # Display current board
            self.display()
            
            # Determine which AI is moving
            current_ai = self.white_ai if self.board.turn == chess.WHITE else self.black_ai
            current_temp = white_temp if self.board.turn == chess.WHITE else black_temp
            
            print(f"{'White' if self.board.turn == chess.WHITE else 'Black'}'s turn (AI)")
            
            # Get AI move
            uci_move = current_ai.predict_full_move_with_sampling(
                self.board.fen(), temperature=current_temp, verbose=verbose
            )
            
            if uci_move:
                print(f"AI plays: {uci_move}")
                self.make_move(uci_move)
                move_count += 1
            else:
                print("AI failed to make a move.")
                break
                
        # Final display and results same as play_game method
        self.display()
        
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            print(f"{winner} wins by checkmate!")
        elif self.board.is_stalemate():
            print("Game drawn by stalemate!")
        elif self.board.is_insufficient_material():
            print("Game drawn due to insufficient material!")
        elif self.board.is_fifty_moves():
            print("Game drawn by fifty-move rule!")
        elif self.board.is_repetition():
            print("Game drawn by repetition!")
        elif move_count >= max_moves:
            print(f"Game stopped after {max_moves} moves.")
        else:
            print("Game ended.")
            
        return {
            'result': self.board.result(),
            'moves': self.move_history,
            'final_fen': self.board.fen(),
            'move_count': move_count
        }


def setup_game_from_notebook(model=None, token_to_id=None, id_to_token=None, 
                            ai_white=True, ai_black=True, 
                            fen=None):
    """
    Set up a game from a Jupyter notebook with the provided model and token mappings.
    
    Args:
        model: ChessModel instance
        token_to_id: Dictionary mapping tokens to IDs
        id_to_token: Dictionary mapping IDs to tokens
        ai_white: Whether white player should be AI
        ai_black: Whether black player should be AI
        fen: Optional starting position
        
    Returns:
        ChessGame instance ready to play
    """
    # Create AI players as needed
    white_ai = ChessAI(model, token_to_id, id_to_token) if ai_white else None
    black_ai = ChessAI(model, token_to_id, id_to_token) if ai_black else None
    
    # Create and return game
    game = ChessGame(fen=fen, white_ai=white_ai, black_ai=black_ai)
    return game
