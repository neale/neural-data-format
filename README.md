
# Neural Data Format: Image Compression and Retrieval with Neural Fields

<div align="center">
<img src="https://raw.githubusercontent.com/neale/neural-canvas/main/neural_canvas/assets/montage.png" alt="logo"></img>
</div>

# Overview (WIP)

The neural data format is a compression algorithm that uses implicit data representations i.e. neural networks to compress images into a small set of weights. Images compressed into .ndf files are smaller on disk and display fewer artifacts than other lossy compression schemes like JPEG. The neural data format is also a continuous representation, meaning that it is resolution agnostic. Once the .ndf is generated, we can decode the file into any arbitrary resoltuion. 

The [neural canvas](https://github.com/neale/neural-canvas) project is used as a backend for the neural network components. 

## Installation

Just clone and pip install

`git clone https://github.com/neale/neural-data-format`

`pip install .`

## Single image compression

**Why would we want to do this?** 

Implicit data representations are cheap! There are less parameters in the neural networks used to represent the 2D and 3D data, than there are pixels or voxels in the data itself. 

Furthermore, the neural representation is flexible, and can be used to extend the data in a number of ways. 

For example, we can instantiate the following function to fit an image

```python
from neural_canvas.models import INRF2D
from neural_canvas.utils import load_image_as_tensor
import numpy as np

img = load_image_as_tensor('neural_canvas/assets/logo.jpg')[0]
model = INRF2D(device='cpu') # or 'cuda'
model.init_map_fn(activations='GELU', weight_init='dip', graph_topology='conv', final_activation='tanh') # better params for fitting
model.fit(img)  # returns a implicit neural representation of the image

print (model.size)  # return size of neural representation
# >> 30083
print (np.prod(img.shape))
# >> 196608

# get original data
img_original = model.generate(output_shape=(256, 256), sample_latent=True)
print ('original size', img_original.shape)
# >> (1, 256, 256, 3)

img_super_res = model.generate(output_shape=(1024,1024), sample_latent=True) 
print ('super res size', img_super_res.shape)
# >> (1, 1024, 1024, 3)
```
## Dataset image compresion

We can compress datasets and store each image as a `.ndf` file. `.ndf` files can be loaded with the `neural_data_format.DiscreteDataset` class. 

### Performance on the MNIST dataset

Convert the MNIST dataset to NDF with `python neural_data_format/neural_mnist.py`.

Train a LeNet classifier for on the NDF data, and compare with the performance of the same classifier trained on the original images.

Training setup:
* LeNet
* Adam optimizer (lr=1e-3)
* Cross entropy
* 5 epochs

| Dataset | Train Loss | Test Accuracy | PSNR   |
|---------|------------|---------------|--------|
| Image   | 0.00       | 97.58         | --     |
| NDF     | 0.03       | 97.91         | 46.144 |

Download the NDF dataset: [Google Drive Link]()

### Performance on the CIFAR10 dataset

ResNet18 clasifier trained for 100 epochs on the NDF data compared to training with the same setup on the original images.

Training setup:
* ResNet18
* Cosine Annealing
* Adam optimizer (lr=1e-3)
* Cross entropy
* 100 epochs

| Dataset | Train Loss | Test Accuracy | PSNR   |
|---------|------------|---------------|--------|
| Image   | 0.00       | 86.39         | --     |
| NDF     | 0.00       | 86.34         | 41.688 |

## Performance comparison with JPEG



## Contributions

Contributions are welcome! If you would like to contribute, please fork the project on GitHub and submit a pull request with your changes.

### Dependency Management

This project uses [Poetry](https://python-poetry.org/) to do environment management. If you want to develop on this project, the best first start is to use Poetry from the beginning. 

To install dependencies (including dev dependencies!) with Poetry:
```bash
poetry shell && poetry install 
```

## License

Released under the MIT License. See the LICENSE file for more details.