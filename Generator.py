import torch
from torch import nn
import VGG16

class Generator(nn.Module):
    def __init__(self,img_dim=3,hidden_layer=64):
        super(Generator,self).__init__()
        layers = [self.make_resnet_block(hidden_layer*4) for _ in range(8)]
        self.gen = nn.Sequential(
            self.make_gen_block(img_dim, hidden_layer),
            self.make_gen_block(hidden_layer, hidden_layer*2),
            self.make_gen_block(hidden_layer*2, hidden_layer*4),
            *layers,
            self.make_gen_block(hidden_layer*4,img_dim,final_layer=True)
            )
        
    def make_gen_block(self,input_dim,output_dim,kernel_size=3,stride=1,padding=2,final_layer=False):
        
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size,stride),
                nn.BatchNorm2d(output_dim),
                nn.ReLU()
                )
        else:
            return nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size,stride),
                nn.Sigmoid()
                )
        
    def make_resnet_block(self,channels,kernel_size,stride):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size,stride),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size,stride),
            nn.ReLU()
            )
        
    def perceptual_false(self,real_img,fake_img,model,device):
        
        real_feat = [feat for feat in model(real_img)]
        fake_feat = [feat for feat in model(fake_img)]
        
        loss = 0
        
        for real_features,fake_features in zip(real_feat,fake_feat):
            loss += nn.functional.mse_loss(real_features,fake_features)
        
        return loss
        
    def get_gen_loss(self,gen,disc,criterion,blur_img_size,real,device,alfa=1e-5):
        
        model = VGG16("vgg16.pth").to(device).eval()
        
        fake = gen(blur_img_size)
        fake_pred = disc(fake)
        fake_target = torch.ones_like(fake_pred).to(device)
        gen_loss = criterion(fake_pred,fake_target)
        
        perceptual_loss_value = self.perceptual_false(real,fake,model,device)
        total_gen_loss = gen_loss + alfa * perceptual_loss_value
        
        return total_gen_loss
        
    def forward(self,x):
        return self.gen(x)


