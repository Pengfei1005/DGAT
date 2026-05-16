import torch
from torch import nn
import torchvision.models as models
from .segformer import Segformer_baseline
from torchvision.transforms import Resize
from transformers import ViTModel, ViTConfig

class Extractor(nn.Module):

    def __init__(self, network):
        super().__init__()
        self.network = network
        if self.network=='resnet101':  #2048,1/32H,1/32W
            cnn = models.resnet101(pretrained=True)  
            modules = list(cnn.children())[:-2]
            self.cnn = nn.Sequential(*modules)
        elif self.network =='vit16': #768,1/16H,1/16W
            config = ViTConfig.from_pretrained("/mnt/data0/qpf_4000/DGAT/models/vit")
            self.vit = ViTModel.from_pretrained("/mnt/data0/qpf_4000/DGAT/models/vit", config=config)
        # Related vit weights are too large and can be downloaded from hugging face official website
        elif 'segformer' in self.network: #512,1/32H,1/32W
            self.cnn = Segformer_baseline(backbone=self.network.split('-')[-1])

        self.fine_tuning()

    def forward(self, imageA, imageB):
        
        if 'resnet' in self.network:
            featA = self.cnn(imageA)  
            featB = self.cnn(imageB)  

        elif 'vit' in self.network:
            torch_resize = Resize([224, 224])
            imageA = torch_resize(imageA)
            imageB = torch_resize(imageB)

            outputsA = self.vit(pixel_values=imageA)
            last_hidden_stateA = outputsA.last_hidden_state
            batch_size, seq_len, hidden_size = last_hidden_stateA.shape
            feature_mapA = last_hidden_stateA[:, 1:, :].contiguous().view(batch_size, 14, 14, hidden_size)
            featA = feature_mapA.permute(0, 3, 1, 2)  # b c h w

            outputsB = self.vit(pixel_values=imageB)
            last_hidden_stateB = outputsB.last_hidden_state
            feature_mapB = last_hidden_stateB[:, 1:, :].contiguous().view(batch_size, 14, 14, hidden_size)
            featB = feature_mapB.permute(0, 3, 1, 2) 

        else:
            featA_list, featA = self.cnn.segformer.stage_123(imageA)
            featB_list, featB = self.cnn.segformer.stage_123(imageB)
            featA = self.cnn.segformer.stage_4(featA)
            featB = self.cnn.segformer.stage_4(featB)
        return featA, featB

    def fine_tuning(self, fine_tune=True):
       
        if 'segformer' in self.network:
            for p in self.cnn.parameters():
                p.requires_grad = fine_tune
        
        elif 'vit' in self.network:
            for param in self.vit.parameters():
                param.requires_grad = True
                
        else:
            for p in self.cnn.parameters():
                p.requires_grad = False
            for c in list(self.cnn.children())[5:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune