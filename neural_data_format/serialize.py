import os
import torch
import numpy as np
from neural_canvas.models.inrf import INRF2D


def compare_generated_images(x, model):
    import matplotlib.pyplot as plt
    c, h, w = x.shape
    gen_data = model.generate(output_shape=(h, w),
                              sample_latent=True)[0]
    fig, axes = plt.subplots(1, 2)
    print (x.shape, gen_data.shape)
    print (x.min(), x.max(), gen_data.min(), gen_data.max())
    axes[0].imshow(np.transpose(x.cpu(), (1, 2, 0)))
    axes[1].imshow(gen_data/255.)
    plt.show()


def save(model, path):
    torch.save(model.state_dict(), path)
    print (model.state_dict().keys())


def load(path, map_location='cpu'):
    model = INRF2D(device=map_location)
    model.load_state_dict(torch.load(path))
    return model


def func(x, model, loss):
    assert x.ndim == 3
    c, h, w = x.shape
    _, _, loss = model.fit(target=x,
              n_iters=100,
              test_resolution=(h,w,c),
              loss=loss)
    #compare_generated_images(x, model)

    state = {k: v.cpu() for k, v in model.map_fn.state_dict().items()}
    return state


def convert_torch_dataset(dataset, device, inrf, loss):
    """Converts a torch.utils.data.Dataset object into a DiscreteDataset"""
    import tqdm

    implicit_dataset = []
    for batch_idx, (data, labels) in enumerate(tqdm.tqdm(dataset)):
        data = data.to(device) #* 2 - 1
        params = [func(x, inrf, loss) for x in data]
        ndf_data = [{'state': p, 'label': l} for p, l in zip(params, labels)]
        implicit_dataset.extend(ndf_data)
        inrf.init_map_weights()
        #if batch_idx > 8:
        #    break

    return implicit_dataset


class DiscreteDataset(torch.utils.data.Dataset):
    """ The discrete dataset assumes that we have an .ndf file for each data point
        We will load all .ndf files in the root directory, and use them as the dataset

        Args:
            root_dir (str): root directory of the dataset
            inrf (INRF2D): the INRF2D model that we will use to generate the data
            data_shape (tuple): the shape of the data that we will generate
    """
    def __init__(self, root_dir, inrf, data_shape):
        self.root_dir = root_dir
        self.data = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.ndf'):
                    self.data.append(os.path.join(root, file))
        self.inrf = inrf
        self.data_shape = data_shape

    def __getitem__(self, idx):
        data = torch.load(self.data[idx])
        label = data['label']
        state = data['state']

        self.inrf.map_fn.load_state_dict(state)
        data = self.inrf.generate(output_shape=self.data_shape, sample_latent=True)[0]
        data = data.transpose(2, 0, 1)  # torch CHW format
        data = data / 255.
        return data, label
    
    def __len__(self):
        return len(self.data)
