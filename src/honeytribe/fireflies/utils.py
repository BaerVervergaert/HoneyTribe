import sklearn
import torch

class TorchAdapter(sklearn.base.BaseEstimator):
    def __init__(self, model: torch.nn.Module, optim, criterion, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.optim = optim
        self.criterion = criterion

    def fit(self, X, y, epochs: int = 10, batch_size: int = 32):
        self.model.train()
        optimizer = self.optim(self.model.parameters())
        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                if hasattr(self.model, 'update'):
                    self.model.update(loss)
        return self

    def predict(self, X) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            predictions = self.model(X_tensor)
        return predictions.cpu().numpy()