import torch 
import numpy as np
from torch import nn
from torch.nn import functional as F
import itertools



class initconv(nn.Module):
    def __init__(self ,in_channels , dim):
        super(initconv,self).__init__()
        self.init_conv = nn.Conv2d(in_channels , dim , kernel_size= 3, stride=2  ,padding =1)
        self.pool = nn.MaxPool2d(kernel_size =2 , stride = 2)
        self.norm = nn.InstanceNorm2d(dim , affine = True ,  momentum =  0.3)
        self.relu = nn.ReLU()
        self.initlayer = nn.Sequential(self.init_conv ,self.norm,self.pool , self.relu)
    def forward(self, img):
        out = self.initlayer(img)
        return out

class convblock_init(nn.Module):
    def __init__(self,in_channels , out ,  num_layers=2):
        super(convblock_init,self).__init__()
        self.conv = nn.Conv2d(out, out , kernel_size = 3,stride =1, padding =1)
        self.norm = nn.InstanceNorm2d(out , momentum =  0.3)
        self.relu  =nn.ELU()
        self.convblock = nn.Sequential(self.conv,self.norm,self.relu)
        self.numlayers = num_layers
        self.initconv  = initconv(in_channels , out)
    def forward(self , img):
        x = self.initconv(img)
        outs = [x]
        count = 0
        res_idx = 0
        for i in range(self.numlayers):
            if count  != 2:
                x =self.convblock(x)
                outs.append(x)
                count+=1
            else:
                x += outs[res_idx]
                x = self.conv(x)
                outs.append(x)
                count = 0
                res_idx+=2

        residual = outs[-1]
        skips = outs     #for decoder

        return residual , skips

class res_block(nn.Module):
    def __init__(self , in_channel , out , num_layers=2):
        super(res_block,self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out , kernel_size = 3,stride =2 ,padding =1)
        self.conv2 = nn.Conv2d(out, out , kernel_size = 3,stride =1 , padding =1)
        self.relu  =nn.ELU()
        self.convblock1 = nn.Sequential(self.conv1, self.relu)
        self.convblock2 = nn.Sequential(self.conv2,self.relu)
        self.numlayers = num_layers
        self.conv_map = nn.Conv2d(in_channel , out , kernel_size= 1 , stride =2)
    def forward(self , ftrs):
        x = self.convblock1(ftrs)
        ftrs_mapped = self.conv_map(ftrs)
        outs = [ftrs_mapped ,x]
        count = 0
        res_idx = 0
        for i in range(self.numlayers- 1):   ######
            if count  != 2:
                x =self.convblock2(x)
                outs.append(x)
                count+=1
            else:
                x += outs[res_idx]
                x = self.convblock2(x)
                outs.append(x)
                count = 0
                res_idx+=2

        residual = outs[-1]
        skips = outs[1:]    #for decoder

        return residual , skips

class net(nn.Module):
    def __init__(self , encoder: str , in_channels =3 ,num_layers = 2):
        super(net , self).__init__()
        if encoder in ["resnet18" , "resnet34"]:
            self.dim = [64 , 128 , 256 , 512]
        elif encoder in ["resnet50" ,  "resnet101"]:
            self.dim = [64 , 128 , 256 , 512 , 1024 ]
        else:
            raise TypeError(f"encoder type {encoder} not valid")

        self.conv1 = convblock_init(in_channels , self.dim[0] , num_layers = num_layers)
        self.convs =  nn.ModuleList([])
        for i in range (1 ,len(self.dim)):
            self.conv = res_block(self.dim[i-1] , self.dim[i] , num_layers = num_layers)
            self.convs.append(self.conv)


    def forward(self,  img):
        r ,s =  self.conv1(img)
        cache = [s]
        for idx , layer in enumerate(self.convs):
            r , s = layer(r)
            cache.append(s)
        cache =  list(itertools.chain(*cache))    #contains skip connections for unet
        return r , cache


############
class deconv_layer(nn.Module):
    def __init__(self ,  in_channels , out_channels):
        super(deconv_layer , self).__init__()
        self.conv1 = nn.Conv2d(in_channels , out_channels , kernel_size = 3 , padding = 1 )
        self.conv2 = nn.Conv2d(out_channels * 2  , out_channels , kernel_size = 3 , padding =1)
    def forward(self , s , p):
        x = F.relu(self.conv1(p))
        upconv = F.interpolate(x , scale_factor = [2,2] , mode='bilinear', align_corners=True )
        _,c,h,w = upconv.size()
        s = F.interpolate(s , size = [h,w], mode = "bilinear" ,align_corners =  True)
        ftr_map = torch.cat((s ,upconv) ,  dim =1)
        out = F.relu(self.conv2(ftr_map))

        return out

class decode(nn.Module):
    def __init__(self,encoder: str, num_layers , out_channels=3 ):
        super(decode,self).__init__()
        if encoder in ["resnet18" , "resnet34"]:
            self.dim = [64 , 128 , 256 , 512]
        elif encoder in ["resnet50" ,  "resnet101"]:
            self.dim = [64 , 128 , 256 , 512 , 1024 ]
        else:
            raise TypeError(f"encoder type {encoder} not valid")
        self.out_channels = out_channels
        self.num_layers  =  num_layers
        self.dim.reverse()
        self.deconvs =  nn.ModuleList([])
        for idx in range(len(self.dim)-1):
            for _ in range (self.num_layers-1):
                self.deconv = deconv_layer(self.dim[idx] , self.dim[idx])
                self.deconvs.append(self.deconv)
            self.deconv = deconv_layer(self.dim[idx] , self.dim[idx+1])
            self.deconvs.append(self.deconv)
        self.residual = nn.Sequential(nn.ConvTranspose2d(self.dim[0] ,out_channels,1))
        self.final_conv = nn.Sequential(nn.ConvTranspose2d(64 , 32 , 1) ,nn.ConvTranspose2d(32 , out_channels , 1))
    def forward(self, x: torch.Tensor ,  skips:list ):
        skips.reverse()
        skips_idx = 0
        residual = self.residual(x)

        for idx , layer in enumerate(self.deconvs):
            skips_idx +=1
            x = layer(skips[skips_idx] , x)
        x = self.final_conv(x)
        _,_,h,w = x.size()
        residual = F.interpolate(residual , [h,w] , mode = "bilinear" , align_corners = True)
        return x+residual


class unet(nn.Module):
    def __init__(self,encoder = "resnet18", num_layers = 2 , in_channels =3, out_channels=3 , mode:str = None): 
        """
        encoder: defaults to resnet18 with 2 blocks each of each layer
        in_channels : 3 or 1 for grayscale image
        out_channels: channel of out image or if for semantic segmentation no_of classes
        mode: None or Segmentation
        """
        super(unet,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder = net(encoder ,  in_channels , num_layers)
        self.decoder = decode(encoder , num_layers , out_channels)

    def forward(self , img):
        x , skips = self.encoder(img)
        out = self.decoder(x , skips)
        out = torch.sigmoid(out) if self.out_channels == 1 else out

        if mode not in ["Segmentation" , None]:
            raise ValueError("mode can only be 'Segmentation' or Default None")
            if mode == "Segmentation":
                out = torch.argmax(out , dim =  1)
            else:
                pass
        return out 





