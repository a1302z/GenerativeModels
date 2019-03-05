import torch

def train(data, model, loss_f, optimizer='SGD', epochs=20):
    for epoch in range(epochs):
        for data, target in data:
            prediction = model(data)
            loss = loss_f(prediction, target)
            loss.backward()
            
