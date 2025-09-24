import sklearn
import torch

class TorchAdapter(sklearn.base.BaseEstimator):
    def __init__(self, model: torch.nn.Module, optim, criterion, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.optim = optim
        self.criterion = criterion
        self.is_fitted = False

    def setup_dataloader(self, X, y, batch_size):
        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    def setup_optimizer(self):
        return self.optim(self.model.parameters())

    def batch_step(self, batch_X, batch_y, optimizer):
        optimizer.zero_grad()
        outputs = self.model(batch_X)
        loss = self.criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        if hasattr(self.model, 'update'):
            self.model.update(loss)
        return outputs, loss

    def partial_fit(self, X, y, batch_size: int = 32, dataloader=None, optimizer=None) -> 'TorchAdapter':
        dataloader = dataloader or self.setup_dataloader(X, y, batch_size)
        optimizer = optimizer or self.setup_optimizer()

        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            self.batch_step(batch_X, batch_y, optimizer)
        return self

    def fit(self, X, y, epochs: int = 10, batch_size: int = 32, max_iterations=None) -> 'TorchAdapter':
        self.model.train()
        optimizer = self.setup_optimizer()
        dataloader = self.setup_dataloader(X, y, batch_size)
        epochs = epochs if max_iterations is None else (max_iterations // len(dataloader) + 1)

        for epoch in range(epochs):
            self.partial_fit(X, y, batch_size=batch_size, dataloader=dataloader, optimizer=optimizer)
        self.is_fitted = True
        return self

    def predict(self, X) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            predictions = self.model(X_tensor)
        return predictions.cpu().numpy()