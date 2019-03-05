import torch

def train(data, model, loss_fn, config):
    if torch.cuda.is_available():
        print("CUDA available")
        model.cuda()
    if config['optimizer'] == 'SGD':
        optim = torch.optim.SGD(list(model.parameters()), lr=float(config['lr']))
    epochs = int(config['epochs'])
    for epoch in range(epochs):
        print("Starting Epoch %d/%d"%(epoch,epochs))
        for i, (data, target) in enumerate(data):
            if i % len(data)/100:
                print('.',end='')
            prediction = model(data)
            loss = loss_fn(prediction, data)
            loss.backward()
            optim.step()
            
    return optim
            
