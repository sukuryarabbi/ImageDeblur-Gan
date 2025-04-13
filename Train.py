import torch
import Generator as g
import Discriminator as d
import DatasetLoader as load
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import matplotlib.pyplot as plt

def Train():
    
    blur_dir = ""
    real_dir = ""
    n_epochs = 100
    lr = 1e-5
    betas = (0.5,0.999)
    display_step = 50
    cur_step = 0
    mean_gen_loss = 0
    mean_disc_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    gen = g.Generator().to(device)
    disc = d.Discriminator().to(device)
    gen_opt = torch.optim.Adam(gen.parameters(),lr,betas)
    disc_opt = torch.optim.Adam(disc.parameters(),lr,betas)

    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
        
    dataset = load.ImageDataSet(blur_dir,real_dir,transform)
    dataloader = DataLoader(dataset,batch_size=32,shuffle=True)

    for epoch in range(n_epochs):
        for batch_idx,(blur_img,sharp_img) in enumerate(dataloader):
            
            gen_opt.zero_grad()
            gen_loss = gen.get_gen_loss(gen,disc,criterion,blur_img,sharp_img,device)
            gen_loss.backward()
            gen_opt.step()
            mean_gen_loss += gen_loss.item() / display_step
            
            disc_opt.zero_grad()
            disc_loss = disc.get_disc_loss(gen,disc,blur_img,sharp_img,criterion,device)
            disc_loss.backward()
            disc_opt.step()
            mean_disc_loss += disc_loss.item() / display_step
            
            if (cur_step % display_step == 0 and cur_step > 0) or epoch == n_epochs:
                print("mean generator loss : {}  mean discriminator loss : {}".format(mean_gen_loss,mean_disc_loss))
                mean_gen_loss = 0
                mean_disc_loss = 0
                
            cur_step += 1
            
            
            