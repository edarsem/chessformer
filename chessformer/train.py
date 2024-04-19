import time

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from dataset import ChessDataset, collate_fn
from model import ChessModel

def train(model, data_loader, device, epochs=2, max_steps=1e4):
    model.train()
    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = CrossEntropyLoss()
    time_start = time.time()

    print(f"Training on {device}")
    steps = 0
    for epoch in range(epochs):
        for batch in data_loader:
            if steps >= max_steps:
                print(f'Training completed: Maximum steps reached ({max_steps} steps)')
                break
            
            sequence_tokens = batch['sequence_tokens'].to(device)
            square_tokens = batch['square_tokens'].to(device)
            meta_lengths = batch['meta_lengths'].to(device)
            next_move = batch['next_move'].to(device)

            optimizer.zero_grad()
            output = model(sequence_tokens, square_tokens, meta_lengths)
            loss = criterion(output, next_move)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, Step {steps+1}, Loss: {loss.item():.4f}')
            steps += 1

    print(f'Time per epoch: {(time.time() - time_start) / epochs:.2f} seconds')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    dataset = ChessDataset('tmp/tokenized_xfen.txt')
    data_loader = DataLoader(dataset, batch_size=1024, shuffle=True, collate_fn=collate_fn)
    
    model = ChessModel()
    train(model, data_loader, device, epochs=2)
