import torch

def train(dataset, model, loss_fn, config):
    print("Learning rate is %f"%float(config['lr']))
    if torch.cuda.is_available():
        print("CUDA available")
        model.cuda()
    if config['optimizer'] == 'SGD':
        optim = torch.optim.SGD(list(model.parameters()), lr=float(config['lr']))
    elif config['optimizer'] == 'Adam':
        optim = torch.optim.Adam(list(model.parameters()), lr=float(config['lr']))
    else:
        print('Unknown optimizer!')
    epochs = int(config['epochs'])
    for epoch in range(epochs):
        print("Starting Epoch %d/%d"%(epoch,epochs))
        for i, (data, target) in enumerate(dataset):
            prediction = model(data)
            loss = loss_fn(prediction, data)
            loss.backward()
            optim.step()
            if i % (len(dataset)/10) == 0:
                print("  Loss after %d/%d iterations: %f"%(i,len(dataset),loss))
        
            
    return optim
            
