import torch
import torch.nn as nn
import torch.optim as optim

class GeneralNN(nn.Module):
    def __init__(self, layer_sizes, activation_fn=nn.ReLU):
        super(GeneralNN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # No activation after the last layer
                self.layers.append(activation_fn())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def train_model(self, train_loader, criterion, optimizer, num_epochs=5, early_stopping_patience=3):
        best_loss = float('inf')
        patience_counter = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                self.save_model('best_model.pth')
            else:
                patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break

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

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
