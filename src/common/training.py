
import torch
import torch.nn as nn
from IPython.display import clear_output

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

_BCE_logits_mean = nn.BCEWithLogitsLoss(reduction="mean")
_BCE_logits_sum = nn.BCEWithLogitsLoss(reduction="sum")


def train_loop(train_dataloader, model, optimizer, val_dataloader = None, num_epochs = 10, verbose = False):
    train_loss_history = []
    train_accuracy_history = []

    test_loss_history = []
    test_accuracy_history = []

    best_val_acc = 0.0
    best_epoch = None
    best_params = None
    try:
        for epoch in range(num_epochs):
            epoch_loss, epoch_accuracy = train_one_epoch(train_dataloader, model, optimizer)

            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"Train loss: {epoch_loss:>7f}, accuracy: {100*epoch_accuracy:>0.2f}%")

            train_loss_history.append(epoch_loss)
            train_accuracy_history.append(epoch_accuracy)

            if val_dataloader:
                epoch_loss, epoch_accuracy = test_model(val_dataloader, model, verbose=False)

                if verbose:
                    print(f"Val loss:  {epoch_loss:>7f}, accuracy: {100*epoch_accuracy:>0.2f}%")

                test_loss_history.append(epoch_loss)
                test_accuracy_history.append(epoch_accuracy)

                if best_val_acc < epoch_accuracy:
                    best_epoch = epoch + 1
                    best_val_acc = epoch_accuracy
                    best_params = [p.detach().cpu() for p in model.parameters()]
            
            if verbose:
                print("-----------------------------")
    except KeyboardInterrupt:
        pass

    if best_params is not None:
        if verbose:
            print(f"\nLoading best params on validation set (epoch {best_epoch}, accuracy: {100*best_val_acc:>0.2f}%)\n")
        with torch.no_grad():
            for param, best_param in zip(model.parameters(), best_params):
                param[...] = best_param
        

    history = {'train_loss': train_loss_history, 'train_accuracy': train_accuracy_history}

    if val_dataloader:
        history['val_loss'] = test_loss_history
        history['val_accuracy'] = test_accuracy_history

    return history



def train_one_epoch(dataloader, model, optimizer, batch_log = 1000):
    model.train()
    epoch_loss = 0
    correct = 0

    n_batches = len(dataloader)

    for i, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        out = model(X)
        loss = _BCE_logits_mean(out, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = (out > 0)
        correct += (pred == y).sum().item()
        epoch_loss += loss.item() * y.shape[0]

        if (i+1) % batch_log == 0:
            print(f"[{i+1}/{n_batches}] loss: {loss.item()}")
        

    size = len(dataloader.dataset)

    return epoch_loss/size, correct/size



def test_model(dataloader, model, verbose = True):
    size = len(dataloader.dataset)
    model.eval()

    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")

    test_loss, correct = 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            out = model(X)
            test_loss += loss_fn(out, y).item()
            pred = (out > 0).int()
            correct += (pred == y).sum().item()

        if verbose:
            print(f"Test Error: Accuracy: {(100*correct/size):>0.2f}%, Avg loss: {test_loss/size:>8f}")

    if not verbose:
        return test_loss/size, correct/size


def test_high_confidence(dataloader, model, low_boundary = 0.3, high_boundary = None, test_accuracy = True):
    size = len(dataloader.dataset)
    model.eval()

    if high_boundary == None:
        high_boundary = 1-low_boundary

    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")

    test_loss, correct = 0, 0

    high_conf_count = 0

    with torch.inference_mode():
        for X,y in dataloader:
            X = X.to(device)
            y = y.to(device)

            out = model(X)

            sigmoid_out = torch.sigmoid(out)

            high_conf_out = (sigmoid_out < low_boundary) | (sigmoid_out > high_boundary)

            high_conf_count += high_conf_out.sum()

            if test_accuracy:
                pred = torch.round(sigmoid_out)

                correct += (high_conf_out * (pred == y)).type(torch.float).sum().item()

    print(f"High confidence samples: {high_conf_count}/{size} = {(100*high_conf_count/size):0.2f}%")

    if test_accuracy:
        correct /= high_conf_count
        print(f"Accuracy for high confidence samples: {(100*correct):>0.1f}%")



def predict(dataloader, model, low_boundary = 0.3, high_boundary = None):
    size = len(dataloader.dataset)
    model.eval()

    batch_size = dataloader.batch_size

    result = torch.zeros((size, 1))

    if high_boundary == None:
        high_boundary = 1-low_boundary


    with torch.no_grad():
        for batch, (X,y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)

            out = model(X)

            sigmoid_out = torch.sigmoid(out)

            high_conf_out = (sigmoid_out < low_boundary) | (sigmoid_out > high_boundary)

            pred = torch.round(sigmoid_out)

            pred.masked_fill_(high_conf_out == False, -1).to('cpu')

            result[batch*batch_size: batch*batch_size+len(X)] = pred


    return result