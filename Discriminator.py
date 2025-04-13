import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self,img_dim=3,hidden_layer=64):
        super(Discriminator(),self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(img_dim, hidden_layer),
            self.make_disc_block(hidden_layer, hidden_layer*2),
            self.make_disc_block(hidden_layer*2, hidden_layer*4),
            self.make_disc_block(hidden_layer*4, hidden_layer*8),
            self.make_disc_block(hidden_layer*8, 1,final_layer=True)
            )
        
    def make_disc_block(self,input_dim,output_dim,kernel_size=4,stride=2,padding=1,final_layer=False):
        
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size,stride,padding),
                nn.BatchNorm2d(output_dim),
                nn.LeakyReLU(0.2)
                )
        else:
            return nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size,stride,padding),
                nn.Sigmoid()
                )
        
    def get_disc_loss(self,gen,disc,blur_img,real_img,criterion,device):
        
        fake_pred = disc(blur_img)
        fake_target = torch.zeros_like(fake_pred).to(device)
        fake_loss = criterion(fake_pred,fake_target)
        
        real_pred = disc(real_img)
        real_target = torch.ones_like(real_pred).to(device)
        real_loss = criterion(real_pred,real_target)
        
        return 0.5*(fake_loss,real_loss)
        
    def forward(self,x):
        return self.disc(x)
        