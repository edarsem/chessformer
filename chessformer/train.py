from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from dataset import ChessDataset, collate_fn
from model import ChessModel

def train(model, data_loader, epochs=2, max_steps=1000):
    model.train()
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()

    steps = 0
    for epoch in range(epochs):
        for batch in data_loader:
            if steps >= max_steps:
                print(f'Training completed: Maximum steps reached ({max_steps} steps)')
                break
            optimizer.zero_grad()
            meta_tokens = batch['meta_tokens']
            pieces_tokens = batch['pieces_tokens']
            squares_tokens = batch['squares_tokens']
            next_move = batch['next_move']
            output = model(meta_tokens, pieces_tokens, squares_tokens)
            loss = criterion(output, next_move)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, Step {steps+1}, Loss: {loss.item():.4f}')
            steps += 1

if __name__ == "__main__":
    dataset = ChessDataset('tmp/tokenized_xfen.txt')
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    model = ChessModel()
    train(model, data_loader)