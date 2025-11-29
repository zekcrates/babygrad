import baby 
import numpy as np 
from baby import nn 

def create_dataset(num_sample: 3000):
    A = np.array([[1, 2], [-0.2, 0.5]])
    mu = np.array([2,1])
    data =np.random.normal(0,1, (num_sample,2)) @ A + mu 
    return data 

def sample_generator(model_generator, num_samples):
    z= baby.Tensor(np.random.normal(0, 1, (num_samples, 2)))
    fake_x = model_generator(z)
    return fake_x.detach().numpy() 

def update_generator(z, model_g, model_d, opt_g , loss_d):
    model_g.train()
    fake_x = model_g(z)
    fake_y= model_d(fake_x)
    for p in model_g.parameters():
        p.requires_grad = True 
    for p in model_d.parameters():
        p.requires_grad =False 

    opt_g.zero_grad() 
    batch_size = z.shape[0]
    ones = baby.Tensor.ones(batch_size)
    
    loss = loss_d(fake_y, ones)
    loss.backward()
    opt_g.step() 

def update_discriminator(x, z, model_g, model_d, opt_d, loss_d):
    model_d.train() 
    fake_x = model_g(z)
    fake_y = model_d(fake_x)
    real_y = model_d(x)

    opt_d.zero_grad()
    batch_size =x.shape[0]
    ones = baby.Tensor.ones(batch_size)
    zeros = baby.Tensor.zeros(batch_size)

    loss = loss_d(real_y,ones)  + loss_d(fake_y, zeros)
    loss.backward()
    opt_d.step() 

def train_gan(data, batch_size, num_epochs):
    for epoch in range(num_epochs):
        begin = (batch_size* epoch) % data.shape[0]
        x = baby.Tensor(data[begin : begin+ batch_size  , : ])
        z = baby.Tensor.randn(batch_size, 2)
        update_discriminator(x,z,generator_model, discriminator_model, discriminator_opt, loss_d)
        update_generator(z, generator_model, discriminator_model,generator_opt, loss_d )
        
generator_model = nn.Linear(2,2)
discriminator_model = nn.Sequential(
    nn.Linear(2,20), 
    nn.ReLU(), 
    nn.Linear(20, 10), 
    nn.ReLU(), 
    nn.Linear(10,2)
)
loss_d = nn.SoftmaxLoss()
generator_opt = baby.optim.Adam(generator_model.parameters() , lr=0.01)
discriminator_opt = baby.optim.Adam(discriminator_model.parameters() , lr=0.01)
data = np.random.normal(0, 1, (3200, 2)) 

train_gan(data, 32, 3000)