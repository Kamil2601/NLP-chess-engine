
import torch
import torch.nn as nn


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def test_model(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()

    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")

    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            out = model(X)
            test_loss += loss_fn(out, y)
            pred = torch.round(torch.sigmoid(out))
            correct += (pred == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def test_high_confidence(dataloader, model, low_boundary = 0.3, high_boundary = None, test_accuracy = True, device = device):
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



def predict(dataloader, model, low_boundary = 0.3, high_boundary = None, device = device):
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