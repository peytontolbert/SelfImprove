import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def train_model(self, train_loader, criterion, optimizer, num_epochs=5):
        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    def evaluate_model(self, test_loader, criterion):
        with torch.no_grad():
            total_loss = 0
            for inputs, labels in test_loader:
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
            print(f'Average Loss: {total_loss / len(test_loader):.4f}')

    def predict(self, inputs):
        with torch.no_grad():
            return self(inputs)
