import torch
import torch.nn as nn

# Define the 3D U-Net architecture
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        out=self.encoder(x)
        return out

class Middel(nn.Module):
    def __init__(self,in_channels):
        super(Middel,self).__init__()
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv3d(in_channels, in_channels*2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_channels*2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_channels*2, in_channels*2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_channels*2),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        out=self.decoder(x)
        return out
class Upsample(nn.Module):
    def __init__(self,channelNum):
        super(Upsample,self).__init__()
        self.decoder =nn.ConvTranspose3d(channelNum,channelNum//2,kernel_size=(1,2,2),stride=(1,2,2))
    def forward(self, x,feature_Map):
        out=self.decoder(x)
        return torch.cat((out,feature_Map),dim=1)
class UNet3D(nn.Module):
    def __init__(self,in_channels,channelNum,out_channels,N_well,well_channel):
        super(UNet3D,self).__init__()
        self.C0=nn.Conv3d(in_channels,channelNum,3,1,1)
        self.C1=ConvBlock(channelNum,channelNum*2)
        self.C2=ConvBlock(channelNum*2,channelNum*4)
        self.C3=ConvBlock(channelNum*4,channelNum*8)
        self.pool=nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.well=nn.Conv3d(N_well,well_channel,3,1,1)
        self.well2=nn.Conv3d(well_channel+channelNum*8,channelNum*8,3,1,1)
        self.globalpool=nn.AdaptiveAvgPool3d((1,1,1))
        self.dense = torch.nn.Linear(in_features=256, out_features=256)
        self.hind = torch.nn.Linear(in_features=256, out_features=256)
        self.out3 = torch.nn.Linear(in_features=256, out_features=7)
        self.M=Middel(channelNum*8)
        self.U1=Upsample(channelNum*8)
        self.C33=ConvBlock(channelNum*8,channelNum*4)
        self.U2=Upsample(channelNum*4)
        self.C22=ConvBlock(channelNum*4,channelNum*2)
        self.U3=Upsample(channelNum*2)
        self.C11=ConvBlock(channelNum*2,channelNum*1)
        self.U4=Upsample(channelNum*2)
        self.out=nn.Conv3d(channelNum,out_channels,3,1,1)
        self.sigm = nn.Sigmoid()
    def forward(self, x,x2):
        d1=self.C0(x)
        d2=self.C1(self.pool(d1))
        d3=self.C2(self.pool(d2))
        d4=self.C3(self.pool(d3))
        x2=self.well(x2)
        d4=torch.cat((d4,x2),dim=1)
        d4=self.well2(d4)
        y2=self.globalpool(d4)
        y2 = torch.flatten(y2, 1)
        y2 = self.dense(y2)
        y2 = self.hind(y2)
        y2 = self.out3(y2)
        u1=self.U1(d4,d3)
        u1=self.C33(u1)
        u2=self.U2(u1,d2)
        u2=self.C22(u2)
        u3=self.U3(u2,d1)
        u3=self.C11(u3)
        out=self.out(u3)
        out=self.sigm(out)
        return out,y2
