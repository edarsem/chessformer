import time

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from dataset import ChessDataset, collate_fn
from model import ChessModel

from config import DEVICE, TRAINING_PARAMS, MODEL_PARAMS, TOKENIZED_XFEN_FILE

def train(model, data_loader, device, epochs=2, max_steps=1e4, lr=1e-3):
    model.train()
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()
    time_start = time.time()

    print(f"Training on {device}")
    steps = 0
    for epoch in range(epochs):
        for batch in data_loader:
            if steps >= max_steps:
                print(f'Training completed: Maximum steps reached ({max_steps} steps)')
                break
            t0 = time.time()

            sequence_tokens = batch['sequence_tokens'].to(device)
            squares_tokens = batch['squares_tokens'].to(device)
            next_move = batch['next_move'].to(device)

            optimizer.zero_grad()
            output = model(sequence_tokens, squares_tokens)
            loss = criterion(output, next_move - model.first_class_token)
            loss.backward()
            optimizer.step()

            dt = time.time() - t0
            tokens_per_sec = sequence_tokens.size(0) / dt
            print(f'Epoch {epoch+1:4d} | Step {steps+1:4d} | Loss: {loss.item():.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}')
            steps += 1

    print(f'Time per epoch: {(time.time() - time_start) / epochs:.2f} seconds')

if __name__ == "__main__":
    dataset = ChessDataset(TOKENIZED_XFEN_FILE)

    data_loader = DataLoader(
        dataset, batch_size=TRAINING_PARAMS['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn
        )
    
    model = ChessModel(
        num_tokens=MODEL_PARAMS['num_tokens'], 
        num_classes=MODEL_PARAMS['num_classes'], 
        d_model=MODEL_PARAMS['d_model'], 
        nhead=MODEL_PARAMS['nhead'], 
        num_layers=MODEL_PARAMS['num_layers'], 
        dropout=MODEL_PARAMS['dropout']
        )
    
    train(model, data_loader, DEVICE, 2, TRAINING_PARAMS['max_steps'], TRAINING_PARAMS['learning_rate'])