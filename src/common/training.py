
import torch
import torch.nn as nn

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train_loop(dataloader, model, optimizer, num_epochs = 10, verbose = False, device = device):
    model.train()
    size = len(dataloader.dataset)
    loss_history = []
    err_rate_history = []

    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    testing_loss_fn = nn.BCEWithLogitsLoss(reduction="sum")

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_err_count = 0

        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            X = X.to(device)
            y = y.to(device)
            out = model(X)
            loss = loss_fn(out, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.inference_mode():
                prediction = torch.round(torch.sigmoid(out))
                errors = prediction != y
                epoch_loss += testing_loss_fn(out, y).item()
                epoch_err_count += errors.sum().item()
        
        if verbose:
            print(f"epoch {epoch+1}/{num_epochs}  loss: {epoch_loss/size:>7f}, err_rate: {100*epoch_err_count/size:>0.2f}%")

        loss_history.append(epoch_loss/size)
        err_rate_history.append(epoch_err_count/size)

    return loss_history, err_rate_history