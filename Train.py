import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

def train(train_loader, model, device, epochs, batch_size, lr, momentum):
    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        for i, data in enumerate(train_loader):
            inputs, output1, output2 = [d.to(device) for d in data]
            optimizer.zero_grad()
            prediction1, prediction2 = model(inputs)
            loss1 = loss_fn(prediction1, output1)
            loss2 = loss_fn(prediction2, output2)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 5 == 4:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {total_loss / 5:.4f}')
                total_loss = 0.0
        duration = time.time() - start_time
    print(f'Finished Training in {duration:.2f} seconds')

    model_dir = './modelos'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    PATH = './modelos/VGG19_model.pth'
    torch.save(model.state_dict(), PATH)
    