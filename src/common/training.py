
import torch
import torch.nn as nn

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

_BCE_logits_mean = nn.BCEWithLogitsLoss(reduction="mean")
_BCE_logits_sum = nn.BCEWithLogitsLoss(reduction="sum")

def train_loop(train_dataloader, model, optimizer, test_dataloader = None, num_epochs = 10, verbose = False):
    train_loss_history = []
    train_accuracy_history = []

    test_loss_history = []
    test_accuracy_history = []

    for epoch in range(num_epochs):
        epoch_loss, epoch_accuracy = train_one_epoch(train_dataloader, model, optimizer)

        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}")
        
        if verbose:
            print(f"Train loss: {epoch_loss:>7f}, accuracy: {100*epoch_accuracy:>0.2f}%")

        train_loss_history.append(epoch_loss)
        train_accuracy_history.append(epoch_accuracy)

        if test_dataloader:
            epoch_loss, epoch_accuracy = test_model(test_dataloader, model, verbose=False)

            if verbose:
                print(f"Test loss:  {epoch_loss:>7f}, accuracy: {100*epoch_accuracy:>0.2f}%")

            test_loss_history.append(epoch_loss)
            test_accuracy_history.append(epoch_accuracy)
        
        if verbose:
            print("-----------------------------")
        

    history = {'train_loss': train_loss_history, 'train_accuracy': train_accuracy_history}

    if test_dataloader:
        history['test_loss'] = test_loss_history
        history['test_accuracy'] = test_accuracy_history

    return history



def train_one_epoch(dataloader, model, optimizer):
    model.train()
    epoch_loss = 0
    correct = 0

    for (X, y) in dataloader:
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)
        out = model(X)
        loss = _BCE_logits_mean(out, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = (out > 0)
        correct += (pred == y).sum().item()
        epoch_loss += loss.item() * y.shape[0]
        
        # if verbose:
        #     print(f"epoch {epoch+1:>5}/{num_epochs}, loss: {epoch_loss/size:>7f}, accuracy: {100*correct/size:>0.2f}%")
    size = len(dataloader.dataset)

    return epoch_loss/size, correct/size



def test_model(dataloader, model, verbose = True):
    size = len(dataloader.dataset)
    model.eval()

    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")

    test_loss, correct = 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

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