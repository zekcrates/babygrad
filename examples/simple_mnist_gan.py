import baby 
import numpy as np 
from baby import nn 
from baby.optim import Adam 
from baby.data import MNISTDataset, DataLoader
from .simple_gan import update_generator, update_discriminator
import baby.optim as optim 


z_dim = 64
image_dim = 28 * 28 
lr = 0.0002 

model_generator = nn.Sequential(
    nn.Linear(z_dim, 256), 
    nn.ReLU(), 
    nn.Linear(256,512), 
    nn.ReLU() ,
    nn.Linear(512,784),
    nn.Tanh()
)

model_discriminator = nn.Sequential(
    nn.Linear(image_dim, 512),
    nn.ReLU(0.2), 
    nn.Linear(512, 256),
    nn.ReLU(0.2),
    nn.Linear(256, 2)


)
opt_g = optim.Adam(model_generator.parameters(), lr=lr)
opt_d = optim.Adam(model_discriminator.parameters(), lr=lr)
loss_d = nn.SoftmaxLoss()

dataset = MNISTDataset("data/train-images-idx3-ubyte.gz", "data/train-labels-idx1-ubyte.gz")
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


for epoch in range(100): 
    d_loss = 0
    g_loss = 0 
    for i, (imgs, _) in enumerate(dataloader):
        
        # Flatten images: [Batch, 1, 28, 28] -> [Batch, 784]
        real_imgs = imgs.view(imgs.size(0), -1) 
        batch_size = real_imgs.shape[0]
        
        z = baby.Tensor.randn(batch_size, z_dim)
        
        d_loss = update_discriminator(real_imgs, z, model_generator, model_discriminator, opt_d, loss_d)
        g_loss = update_generator(z, model_generator, model_discriminator, opt_g, loss_d)
        
    print(f"Epoch {epoch}: D_loss: {d_loss:.4f}, G_loss: {g_loss:.4f}")