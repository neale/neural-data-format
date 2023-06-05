import matplotlib.pyplot as plt
import torch

from neural_canvas.models import INRF2D
from neural_canvas.utils import load_image_as_tensor
import numpy as np

img = load_image_as_tensor('neural_data_format/assets/logo.jpg')[0]
imgs = []
fig, axes = plt.subplots(1, 4, figsize=(20, 4))


for i, width in enumerate([4, 8, 16, 32]):
    model = INRF2D(device='cpu') # or 'cuda'
    model.init_map_fn(activations='ELU',
                      weight_init='dip', 
                      conv_feature_map_size=width,
                      graph_topology='conv', 
                      final_activation='tanh') # better params for fitting
    _, _, loss = model.fit(img, n_iters=5000)  # returns a implicit neural representation of the image
    gen_img = model.generate(output_shape=(256, 256),sample_latent=True) # generate image from the implicit neural representation
    print (model)
    print (f'NDF size: {model.size}')  # return size of neural representation
    print (f'NDF loss: {loss}, L2 Error: {torch.nn.functional.mse_loss(img, gen_img[0]):.2f}')
    # >> 30083
    print ('image size', np.prod(img.shape))
    # >> 196608
    axes[i].imshow(gen_img[0])
    axes[i].set_title(f'width: {width}, loss: {loss:.2f}')

plt.show()

# get original data
img_original = model.generate(output_shape=(256, 256), sample_latent=True)
print ('original size', img_original.shape)
# >> (1, 256, 256, 3)
img_super_res = model.generate(output_shape=(1024,1024), sample_latent=True) 
print ('super res size', img_super_res.shape)
# >> (1, 1024, 1024, 3)