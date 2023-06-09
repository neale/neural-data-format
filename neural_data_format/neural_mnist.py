import os
import tqdm
import torch
import torchvision

from neural_canvas.models.inrf import INRF2D
from neural_canvas.losses import LossModule

from neural_data_format.utils import train_classifier
from neural_data_format.models import LeNet
from neural_data_format.serialize import convert_torch_dataset, DiscreteDataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset_train = torchvision.datasets.MNIST(root='./tmp',
                                     train=True,
                                     download=True, 
                                     transform=torchvision.transforms.ToTensor())
dataset_test = torchvision.datasets.MNIST(root='./tmp',
                                        train=False,
                                        download=True,
                                        transform=torchvision.transforms.ToTensor())

train_loader_img = torch.utils.data.DataLoader(dataset_train, batch_size=128, num_workers=2)
test_loader_img = torch.utils.data.DataLoader(dataset_test, batch_size=128, num_workers=2)

root_dir = 'ndf_datasets/mnist'
loss = LossModule(l2_alpha=1.0, device='cpu')
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
dataset_train_ndf = convert_torch_dataset(train_loader_img, loss=loss, model_params=model_params, device=device)
os.makedirs(os.path.join(root_dir, 'train'), exist_ok=True)
for i, data in enumerate(dataset_train_ndf):
    torch.save(data, os.path.join(root_dir, 'train', f'{i}.ndf'.zfill(7)))

# Create test set (NDF)
dataset_test_ndf = convert_torch_dataset(test_loader_img, loss=loss, model_params=model_params, device=device)
os.makedirs(os.path.join(root_dir, 'test'), exist_ok=True)
for i, data in enumerate(dataset_test_ndf):
    torch.save(data, os.path.join(root_dir, 'test', f'{i}.ndf'.zfill(7)))
    
# load back data
inrf = INRF2D(device='cpu', latent_dim=128)
inrf.c_dim = 1
inrf.init_map_fn(**model_params)

batch = next(iter(train_loader_img))[0]
#inrf.init_map_fn(**inrf_params)

dataset_train_ndf = DiscreteDataset(
    os.path.join(root_dir, 'train'),
    inrf,
    data_shape=batch.shape[2:])
train_loader_ndf = torch.utils.data.DataLoader(dataset_train_ndf, batch_size=64, num_workers=0)

dataset_test_ndf = DiscreteDataset(
    os.path.join(root_dir, 'test'),
    inrf,
    data_shape=batch.shape[2:])
test_loader_ndf = torch.utils.data.DataLoader(dataset_test_ndf, batch_size=64, num_workers=2)

model = LeNet().to(device)

print ('Training classifier on NDF data')
train_classifier(train_loader_ndf, test_loader_ndf, device=device)
print ('Training classifier on original data')
train_classifier(train_loader_img, test_loader_img, device=device)




