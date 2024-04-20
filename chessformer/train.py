import time

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from dataset import ChessDataset, collate_fn
from model import ChessModel

from config import DEVICE, TRAINING_PARAMS, MODEL_PARAMS, TOKENIZED_XFEN_FILE

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
            squares_tokens = batch['squares_tokens'].to(device)
            next_move = batch['next_move'].to(device)

            optimizer.zero_grad()
            output = model(sequence_tokens, squares_tokens)
            loss = criterion(output, next_move - model.first_class_token)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, Step {steps+1}, Loss: {loss.item():.4f}')
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
    
    train(model, data_loader, DEVICE, 2, TRAINING_PARAMS['max_steps'])