import os
import tqdm
import torch
import torchvision

from neural_canvas.models.inrf import INRF2D
from neural_canvas.losses import LossModule

from neural_data_format.utils import train_classifier
from neural_data_format.models import ResNet18
from neural_data_format.serialize import convert_torch_dataset, DiscreteDataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset_train = torchvision.datasets.CIFAR10(
    root='./tmp',
    train=True,
    download=True, 
    transform=torchvision.transforms.ToTensor())
dataset_test = torchvision.datasets.CIFAR10(
    root='./tmp',
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor())

train_loader_img = torch.utils.data.DataLoader(dataset_train, batch_size=32, num_workers=2)
test_loader_img = torch.utils.data.DataLoader(dataset_test, batch_size=32, num_workers=2)

root_dir = 'ndf_datasets/cifar10'

loss = LossModule(l2_alpha=1.0, device=device)#'cpu')

model_params = {
    'graph_topology':'siren',#'conv',
    'input_encoding_dim': 1,#48,
    'conv_feature_map_size': 64,
    'mlp_layer_width': 128,
    'weight_init': 'siren',#'dip',
    'activations': 'siren',#'GELU',
    'final_activation': 'sigmoid',
    'num_fourier_freqs': None,#24
}

# Create training set (NDF)
"""
psnr_train = convert_torch_dataset(
    train_loader_img,
    loss=loss,
    model_params=model_params,
    device=device,
    save_dir=os.path.join(root_dir, 'train'),
    debug=False)

# Create test set (NDF)
psnr_test = convert_torch_dataset(
    test_loader_img,
    loss=loss,
    model_params=model_params,
    save_dir=os.path.join(root_dir, 'test'),
    device=device)
    
print (f'Mean PSNR Train: {psnr_train}')
print (f'Mean PSNR Test: {psnr_test}')
"""
# load back data
inrf = INRF2D(device='cpu', latent_dim=64)
inrf.c_dim = 3
inrf.init_map_fn(**model_params)
batch = next(iter(train_loader_img))[0]

transforms_train = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
])
transforms_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
])
dataset_train_ndf = DiscreteDataset(
    os.path.join(root_dir, 'train'),
    inrf,
    data_shape=batch.shape[2:],
    transforms=transforms_train,)
train_loader_ndf = torch.utils.data.DataLoader(dataset_train_ndf,
                                               batch_size=64,
                                               num_workers=0,
                                               shuffle=True)

dataset_test_ndf = DiscreteDataset(
    os.path.join(root_dir, 'test'),
    inrf,
    data_shape=batch.shape[2:],
    transforms=transforms_test,)
test_loader_ndf = torch.utils.data.DataLoader(dataset_test_ndf,
                                              batch_size=64,
                                              num_workers=8,
                                              shuffle=False)

model = ResNet18().to(device)
print ('Training classifier on NDF data')
train_classifier(model, train_loader_ndf, test_loader_ndf, device=device)
print ('Training classifier on original data')
train_classifier(model, train_loader_img, test_loader_img, device=device)




