import math
import torch
import numpy as np


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def train_classifier(model, train_loader, test_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    for epoch in range(100):
        model.train()
        for data, labels in train_loader:
            #breakpoint()
            data = data.to(device).float()
            labels = labels.to(device).long()
            output = model(data)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print (f'[Epoch {epoch}] Train loss: {loss.item():.2f}')
        if epoch % 2 == 0:
            model.eval()
            correct = 0
            with torch.no_grad():
                for data, labels in test_loader:
                    data = data.to(device).float()
                    labels = labels.to(device).long()
                    output = model(data)
                    loss = criterion(output, labels)
                    correct += (output.argmax(dim=1) == labels).sum().item()
            accuracy = correct / len(test_loader.dataset)
            print (f'[Epoch {epoch}] Test_acc: {(accuracy*100):.4f}') 
        
        scheduler.step()