import os
import tqdm
import torch
import torchvision

from neural_canvas.models.inrf import INRF2D
from neural_canvas.losses import LossModule

from neural_data_format.models import LeNet
from neural_data_format.serialize import convert_torch_dataset, DiscreteDataset


def train_classifier(train_loader, test_loader, device):
    model = LeNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(5):
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
            correct = 0
            for data, labels in test_loader:
                data = data.to(device).float()
                labels = labels.to(device).long()
                output = model(data)
                loss = criterion(output, labels)
                correct += (output.argmax(dim=1) == labels).sum().item()
            accuracy = correct / len(test_loader.dataset)
            print (f'[Epoch {epoch}] Test_acc: {(accuracy*100):.4f}') 


device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset_train = torchvision.datasets.MNIST(root='./tmp',
                                     train=True,
                                     download=True, 
                                     transform=torchvision.transforms.ToTensor())
dataset_test = torchvision.datasets.MNIST(root='./tmp',
                                        train=False,
                                        download=True,
                                        transform=torchvision.transforms.ToTensor())

train_loader_img = torch.utils.data.DataLoader(dataset_train, batch_size=8, num_workers=2)
test_loader_img = torch.utils.data.DataLoader(dataset_test, batch_size=8, num_workers=2)

root_dir = 'ndf_datasets/mnist'
inrf_params = {
    'graph_topology':'conv',
    'input_encoding_dim': 48,
    'conv_feature_map_size': 64,
    'weight_init': 'dip',
    'activations': 'GELU',
    'final_activation': 'sigmoid',
    'num_fourier_freqs': 24
}
inrf = INRF2D(device=device)
batch, _ = next(iter(train_loader_img))
inrf.c_dim = batch.shape[1]
inrf.init_map_fn(**inrf_params)
loss = LossModule(l2_alpha=1.0, device=device)

# Create training set (NDF)
dataset_train_ndf = convert_torch_dataset(train_loader_img, device=device, inrf=inrf, loss=loss)
os.makedirs(os.path.join(root_dir, 'train'), exist_ok=True)
for i, data in enumerate(dataset_train_ndf):
    torch.save(data, os.path.join(root_dir, 'train', f'{i}.ndf'.zfill(7)))
    
# Create test set (NDF)
dataset_test_ndf = convert_torch_dataset(test_loader_img, device=device, inrf=inrf, loss=loss)
os.makedirs(os.path.join(root_dir, 'test'), exist_ok=True)
for i, data in enumerate(dataset_test_ndf):
    torch.save(data, os.path.join(root_dir, 'test', f'{i}.ndf'.zfill(7)))
    
# load back data
inrf = INRF2D(device='cpu')
inrf.c_dim = batch.shape[1]
inrf.init_map_fn(**inrf_params)

dataset_train_ndf = DiscreteDataset(
    os.path.join(root_dir, 'train'),
    inrf,
    data_shape=batch.shape[2:])
train_loader_ndf = torch.utils.data.DataLoader(dataset_train_ndf, batch_size=8, num_workers=2)

dataset_test_ndf = DiscreteDataset(
    os.path.join(root_dir, 'test'),
    inrf,
    data_shape=batch.shape[2:])
test_loader_ndf = torch.utils.data.DataLoader(dataset_test_ndf, batch_size=8, num_workers=2)

print ('Training classifier on NDF data')
train_classifier(train_loader_ndf, test_loader_ndf, device=device)
print ('Training classifier on original data')
train_classifier(train_loader_img, test_loader_img, device=device)



