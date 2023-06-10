import os
import tqdm
import torch
import numpy as np
from neural_canvas.models.inrf import INRF2D
from neural_data_format.utils import psnr


def compare_generated_images(x, model, z, debug=False):
    import matplotlib.pyplot as plt
    c, h, w = x.shape
    gen_data = model.generate(output_shape=(h, w),
                              latents=z)[0]
    x = np.transpose(x.cpu(), (1, 2, 0)).numpy()
    gen_data = gen_data / 255.
    quality = psnr(x, gen_data)
    if debug:
        _, axes = plt.subplots(1, 2)
        print (x.shape, gen_data.shape, x.min(), x.max(), gen_data.min(), gen_data.max())
        if c == 1:
            axes[0].imshow(x, cmap='gray')
            axes[1].imshow(gen_data, cmap='gray')
        else:
            axes[0].imshow(x)
            axes[1].imshow(gen_data)
        plt.suptitle(f'PSNR: {quality:.4f}')
        plt.show()
    return quality


def save(model, path):
    torch.save(model.state_dict(), path)
    print (model.state_dict().keys())


def load(path, map_location='cpu'):
    model = INRF2D(device=map_location)
    model.load_state_dict(torch.load(path))
    return model


def func(x, loss, model_params, device, debug=False):
    assert x.ndim == 3
    c, h, w = x.shape
    device = 'cpu'
    model = INRF2D(device=device, latent_dim=64)
    model.c_dim = c
    model.init_map_fn(**model_params)

    _, _, loss, latent = model.fit(
        target=x,
        n_iters=100,
        test_resolution=(h,w,c),
        loss=loss,
        trainable_latent=True,
    )
    #model.device = 'cpu'
    #model.map_fn = model.map_fn.cpu()
    for k, v in latent.items():
        if isinstance(v, torch.Tensor):
            latent[k] = v.cpu()

    #model.map_fn = torch.quantization.quantize_dynamic(
    #    model.map_fn,
    #    {torch.nn.Linear},
    #    dtype=torch.qint8
    #)
    print (model.map_fn)
    quality = compare_generated_images(x, model, latent, debug=debug)
    state = {k: v.cpu() for k, v in model.map_fn.state_dict().items()}

    return state, latent, quality


def convert_torch_dataset(dataset, loss, model_params, device, save_dir, debug=False):
    """Converts a torch.utils.data.Dataset object into a DiscreteDataset"""
    os.makedirs(save_dir, exist_ok=True)
    psnr_dataset = []
    i = 0
    for batch_idx, (data, labels) in enumerate(tqdm.tqdm(dataset)):
        data = data.to(device) 
        states = [func(x, loss, model_params, device, debug) for x in data]
        ndf = [s[0] for s in states]
        latents = [s[1] for s in states]
        quality = [s[2] for s in states]

        ndf_data = [{'state': t, 'label': y, 'latent': z} for t, y, z in zip(ndf, labels, latents)]
        for data in ndf_data:
            torch.save(data, os.path.join(save_dir, f'{i}.ndf'.zfill(7)))
            i += 1
        psnr_dataset.extend(quality)
        #if batch_idx > 4: break
    return torch.tensor(psnr_dataset).mean()


class DiscreteDataset(torch.utils.data.Dataset):
    """ The discrete dataset assumes that we have an .ndf file for each data point
        We will load all .ndf files in the root directory, and use them as the dataset

        Args:
            root_dir (str): root directory of the dataset
            inrf (INRF2D): the INRF2D model that we will use to generate the data
            data_shape (tuple): the shape of the data that we will generate
    """
    def __init__(self, root_dir, inrf, data_shape, transforms=None):
        self.root_dir = root_dir
        self.data = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.ndf'):
                    self.data.append(os.path.join(root, file))
        self.inrf = inrf
        self.data_shape = data_shape
        self.transforms = transforms

    def __getitem__(self, idx):
        data = torch.load(self.data[idx])
        label = data['label']
        state = data['state']
        latent = data['latent']
        #breakpoint()
        self.inrf.map_fn.load_state_dict(state)
        data = self.inrf.generate(output_shape=self.data_shape, latents=latent)[0]
        if self.transforms is not None:
            data = self.transforms(data)
        data = data / 255.
        return data, label
    
    def __len__(self):
        return len(self.data)
