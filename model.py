import torch
import torch.nn as nn
from decoder.dec_net import MappingNet, AXform
from encoder_dgcnn.dgcnn import DGCNN
from config import params
from torchvision import models
from decoder.utils.utils import *
from propagation import Propagation

p = params()

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.pc_encoder = DGCNN()

        base = models.resnet34(pretrained=False)
        self.layer1 = nn.Sequential(*list(base.children())[:5])
        self.layer2 = nn.Sequential(*list(base.children())[5])
        self.layer3 = nn.Sequential(*list(base.children())[6])
        self.layer4 = nn.Sequential(*list(base.children())[7])

        self.deconv1_1 = nn.ConvTranspose2d(512,256,5,stride=2,padding=2,output_padding=1)
        self.deconv1_2 = nn.ConvTranspose2d(256,128,5,stride=2,padding=2,output_padding=1)
        self.deconv1_3 = nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=1)
        self.deconv1_4 = nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1)

        self.conv6_1 = nn.Conv2d(64,32,3,stride=1,padding=1)
        self.conv6_2 = nn.Conv2d(32,3,3,stride=1,padding=1)

        self.num_branch = p.num_branch
        self.K1 = p.K1
        self.K2 = p.K2
        self.N = p.N
        
        self.fc0_up = nn.Linear(3, 256)
        self.fc0_down = nn.Linear(256, 3)
        self.cross_attn0_0 = nn.MultiheadAttention(
            256, 4, batch_first=True)
        self.layer_norm0_0 = nn.LayerNorm(256)
        self.cross_attn0_1 = nn.MultiheadAttention(
            256, 4, batch_first=True)
        self.layer_norm0_1 = nn.LayerNorm(256)

        self.fc1_up = nn.Linear(3, 128)
        self.fc1_down = nn.Linear(128, 3)
        self.cross_attn1_0 = nn.MultiheadAttention(
            128, 4, batch_first=True)
        self.layer_norm1_0 = nn.LayerNorm(128)
        self.cross_attn1_1 = nn.MultiheadAttention(
            128, 4, batch_first=True)
        self.layer_norm1_1 = nn.LayerNorm(128)

        self.fc2_up = nn.Linear(3, 64)
        self.fc2_down = nn.Linear(64, 3)
        self.cross_attn2_0 = nn.MultiheadAttention(
            64, 4, batch_first=True)
        self.layer_norm2_0 = nn.LayerNorm(64)
        self.cross_attn2_1 = nn.MultiheadAttention(
            64, 4, batch_first=True)
        self.layer_norm2_1 = nn.LayerNorm(64)

        self.featmap = nn.ModuleList([MappingNet(self.K1) for i in range(self.num_branch)])
        self.pointgen = nn.ModuleList([AXform(self.K1, self.K2, self.N) for i in range(self.num_branch)])

        self.propagation = Propagation()

    def forward(self, x_part, view):
        pc_feat = self.pc_encoder(x_part)
        pc_feat = pc_feat.permute(0, 2, 1)
        x_part = x_part.permute(0, 2, 1)

        batch_size = view.shape[0]
        
        view = self.layer1(view)
        view = self.layer2(view)
        view = self.layer3(view)
        view = self.layer4(view)

        x_feat = pc_feat
        x_1 = torch.empty(size=(x_part.shape[0], 0, 3)).to(x_part.device)
        for i in range(self.num_branch):
            _x_1 = self.pointgen[i](self.featmap[i](x_feat)) 
            x_1 = torch.cat((x_1, _x_1), dim=1)

        x_partial = x_part.contiguous()
        x_partial = farthest_point_sample(x_partial, 1024)
        x_1_1024 = farthest_point_sample(x_1, 1024)

        de1_view = self.deconv1_1(view).view(batch_size, 256, -1).permute(0, 2, 1)
        x_1_256 = self.fc0_up(x_1_1024)
        x, _ = self.cross_attn0_0(de1_view, x_1_256, x_1_256)
        de1_view = self.layer_norm0_0(x + de1_view)
        x, _ = self.cross_attn0_1(x_1_256, de1_view, de1_view)
        x_1_256 = self.layer_norm0_1(x + x_1_256)
        x_1_256 = self.fc0_down(x_1_256)
    
        de1_view = de1_view.reshape(batch_size, 256, 14, 14)
        de2_view = self.deconv1_2(de1_view).view(batch_size, 128, -1).permute(0, 2, 1)
        x_1_512 = self.fc1_up(x_1_1024)
        x, _ = self.cross_attn1_0(de2_view, x_1_512, x_1_512)
        de2_view = self.layer_norm1_0(x + de2_view)
        x, _ = self.cross_attn1_1(x_1_512, de2_view, de2_view)
        x_1_512 = self.layer_norm1_1(x + x_1_512)
        x_1_512 = self.fc1_down(x_1_512)

        de2_view = de2_view.reshape(batch_size, 128, 28, 28)
        de3_view = self.deconv1_3(de2_view).view(batch_size, 64, -1).permute(0, 2, 1)
        x_1_1024 = self.fc2_up(x_1_1024)
        x, _ = self.cross_attn2_0(de3_view, x_1_1024, x_1_1024)
        de3_view = self.layer_norm2_0(x + de3_view)
        x, _ = self.cross_attn2_1(x_1_1024, de3_view, de3_view)
        x_1_1024 = self.layer_norm2_1(x + x_1_1024)
        x_1_1024 = self.fc2_down(x_1_1024)

        x_1 = torch.cat((x_1_256, x_1_512, x_1_1024), dim=1)
        x_1 = farthest_point_sample(x_1, 1024)

        final = torch.cat((x_1, x_partial), dim=1)

        de3_view = de3_view.reshape(batch_size, 64, 56, 56)
        de3_view = self.conv6_1(de3_view)
        de3_view = self.conv6_2(de3_view)
        de3_view = de3_view.view(batch_size,56*56,3)
        pc_generate = farthest_point_sample(de3_view, 2048)

        pred_pcds = self.propagation(final.detach(), final.detach())[0]
        complete = farthest_point_sample(pred_pcds.contiguous() , 2048)

        return complete, final, pc_generate, pred_pcds