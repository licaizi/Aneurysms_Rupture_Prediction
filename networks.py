import torch
import torch.nn as nn
import torch.nn.init as init

def init_weights(net, init_type='kaiming', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class Encoder3D(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(Encoder3D, self).__init__()
        self.maxpool = nn.MaxPool3d(2)
        self.activate = nn.ReLU(inplace=True)
        self.en1 = nn.Sequential(
                        nn.Conv3d(1,64,kernel_size=3,padding=1),
                        nn.InstanceNorm3d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(64,64,kernel_size=3,padding=1),
                        nn.InstanceNorm3d(64),
                        nn.ReLU(inplace=True),
                        nn.MaxPool3d(2))
        self.en2 = nn.Sequential(
                        nn.Conv3d(64,128,kernel_size=3,padding=1),
                        nn.InstanceNorm3d(128),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(128,128,kernel_size=3,padding=1),
                        nn.InstanceNorm3d(128),
                        nn.ReLU(inplace=True),
                        nn.MaxPool3d(2))
        self.en3 = nn.Sequential(
                        nn.Conv3d(128,256,kernel_size=3,padding=1),
                        nn.InstanceNorm3d(256),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(256,256,kernel_size=3,padding=1),
                        nn.InstanceNorm3d(256),
                        nn.ReLU(inplace=True),
                        nn.MaxPool3d(2))
        self.en4 = nn.Sequential(
                        nn.Conv3d(256,512,kernel_size=3,padding=1),
                        nn.InstanceNorm3d(512),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(512,512,kernel_size=3,padding=1),
                        nn.InstanceNorm3d(512),
                        nn.ReLU(inplace=True),
                        nn.MaxPool3d(8))
        self.de1 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                        nn.Conv3d(64,64,kernel_size=3,padding=1),
                        nn.InstanceNorm3d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(64,1,kernel_size=3,padding=1),
                        nn.Sigmoid())
        self.de2 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                        nn.Conv3d(128,128,kernel_size=3,padding=1),
                        nn.InstanceNorm3d(128),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(128,64,kernel_size=3,padding=1),
                        nn.InstanceNorm3d(64),
                        nn.ReLU(inplace=True))
        self.de3 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                        nn.Conv3d(256,256,kernel_size=3,padding=1),
                        nn.InstanceNorm3d(256),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(256,128,kernel_size=3,padding=1),
                        nn.InstanceNorm3d(128),
                        nn.ReLU(inplace=True))
        self.de4 = nn.Sequential(
                        nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True),
                        nn.Conv3d(512,512,kernel_size=3,padding=1),
                        nn.InstanceNorm3d(512),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(512,256,kernel_size=3,padding=1),
                        nn.InstanceNorm3d(256),
                        nn.ReLU(inplace=True))
        

    def forward(self,x):
        feat = self.en1(x)
        feat = self.en2(feat)
        feat = self.en3(feat)
        feat = self.en4(feat)
        out = self.de4(feat)
        out = self.de3(out)
        out = self.de2(out)
        out = self.de1(out)
        
        return out, feat

class ResEncoder3D(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(ResEncoder3D, self).__init__()
        self.maxpool = nn.MaxPool3d(2)
        self.finalmaxpool = nn.MaxPool3d(8)
        self.activate = nn.ReLU(inplace=True)
        self.en1 = nn.Sequential(
                        nn.Conv3d(1,64,kernel_size=3,padding=1),
                        nn.InstanceNorm3d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(64,64,kernel_size=3,padding=1),
                        nn.InstanceNorm3d(64))
        self.en2 = nn.Sequential(
                        nn.Conv3d(64,128,kernel_size=3,padding=1),
                        nn.InstanceNorm3d(128),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(128,128,kernel_size=3,padding=1),
                        nn.InstanceNorm3d(128))
        self.en3 = nn.Sequential(
                        nn.Conv3d(128,256,kernel_size=3,padding=1),
                        nn.InstanceNorm3d(256),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(256,256,kernel_size=3,padding=1),
                        nn.InstanceNorm3d(256))
        self.en4 = nn.Sequential(
                        nn.Conv3d(256,512,kernel_size=3,padding=1),
                        nn.InstanceNorm3d(512),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(512,512,kernel_size=3,padding=1),
                        nn.InstanceNorm3d(512))
        self.de1 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                        nn.Conv3d(64,64,kernel_size=3,padding=1),
                        nn.InstanceNorm3d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(64,1,kernel_size=3,padding=1),
                        nn.Sigmoid())
        self.de2 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                        nn.Conv3d(128,128,kernel_size=3,padding=1),
                        nn.InstanceNorm3d(128),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(128,64,kernel_size=3,padding=1),
                        nn.InstanceNorm3d(64),
                        nn.ReLU(inplace=True))
        self.de3 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                        nn.Conv3d(256,256,kernel_size=3,padding=1),
                        nn.InstanceNorm3d(256),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(256,128,kernel_size=3,padding=1),
                        nn.InstanceNorm3d(128),
                        nn.ReLU(inplace=True))
        self.de4 = nn.Sequential(
                        nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True),
                        nn.Conv3d(512,512,kernel_size=3,padding=1),
                        nn.InstanceNorm3d(512),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(512,256,kernel_size=3,padding=1),
                        nn.InstanceNorm3d(256),
                        nn.ReLU(inplace=True))
        

    def forward(self,x):
        feat = self.maxpool(self.activate((self.en1(x) + x)))
        feat = self.maxpool(self.activate((self.en2(feat) + feat)))
        feat = self.maxpool(self.activate((self.en3(feat) + feat)))
        feat = self.finalmaxpool(self.activate((self.en4(feat) + feat)))
        out = self.de4(feat)
        out = self.de3(out)
        out = self.de2(out)
        out = self.de1(out)
        
        return out, feat

class Classifier3D(nn.Module):
    def __init__(self,in_channels):
        super(Classifier3D, self).__init__()
        self.linear_block = nn.Sequential(
                        nn.Linear(in_channels,1),
        )
    def forward(self,x):
        x = torch.flatten(x,start_dim=1)
        x = self.linear_block(x)
    
        return x

class R3D_18_Encoder(nn.Module):
    def __init__(self):
        super(R3D_18_Encoder,self).__init__()
        conv_makers=[Conv3DSimple] * 4
        
        self.feature = 32
        self.inplanes = self.feature
        self.stem = nn.Sequential(nn.Conv3d(1, self.feature, kernel_size=(7, 7, 7), stride=(2, 2, 2),padding=(3, 3, 3), bias=False),
                                  nn.BatchNorm3d(self.feature),
                                  nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(BasicBlock, conv_makers[0], self.feature, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, conv_makers[1], self.feature*2, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, conv_makers[2], self.feature*2, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, conv_makers[3], self.feature*2, 4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.de0 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                        nn.Conv3d(self.feature,self.feature,kernel_size=3,padding=1),
                        nn.BatchNorm3d(self.feature),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(self.feature,1,kernel_size=3,padding=1),
                        nn.Sigmoid()
        )
        self.de1 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                        nn.Conv3d(self.feature,self.feature,kernel_size=3,padding=1),
                        nn.BatchNorm3d(self.feature),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(self.feature,self.feature,kernel_size=3,padding=1),
                        nn.ReLU(inplace=True),
        )
        self.de2 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                        nn.Conv3d(self.feature*2,self.feature,kernel_size=3,padding=1),
                        nn.BatchNorm3d(self.feature),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(self.feature,self.feature,kernel_size=3,padding=1),
                        nn.BatchNorm3d(self.feature),
                        nn.ReLU(inplace=True)
        )
        self.de3 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                        nn.Conv3d(self.feature*2,self.feature*2,kernel_size=3,padding=1),
                        nn.BatchNorm3d(self.feature*2),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(self.feature*2,self.feature*2,kernel_size=3,padding=1),
                        nn.BatchNorm3d(self.feature*2),
                        nn.ReLU(inplace=True)
        )
        self.de4 = nn.Sequential(
                        nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True),
                        nn.Conv3d(self.feature*2,self.feature*2,kernel_size=3,padding=1),
                        nn.BatchNorm3d(self.feature*4),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(self.feature*2,self.feature*2,kernel_size=3,padding=1),
                        nn.BatchNorm3d(self.feature*4),
                        nn.ReLU(inplace=True)
        )

        self._initialize_weights()


    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self,x):
        x = self.stem(x) # 1*64^3 - 64*32^3
        x = self.layer1(x) # 64*32^3 - 64*32^3
        x2 = self.layer2(x) # 64*32^3 - 128*16^3
        x = self.layer3(x2) # 128*16^3 - 256*8^3
        x4 = self.layer4(x) # 256*8^3 - 512*4^3
        feat = self.avgpool(x4) # 512*1^3
        x = self.de4(feat) # 512*4^3 - 256*4^3
        x = self.de3(x) # 256*4^3 - 128*8^3
        x = self.de2(x) # 128*8^3 - 64*16^3
        x = self.de1(x) # 64*16^3 - 64*32^3
        x = self.de0(x) # 64*32^3 - 1*64^3
        return x, feat#  , x2, x4

class R3D_18_Encoder_Split(nn.Module):
    def __init__(self):
        super(R3D_18_Encoder_Split,self).__init__()
        conv_makers=[Conv3DSimple] * 4

        self.out_dim = 64 + 64
        self.feature = 32
        self.inplanes = self.feature
        self.stem = nn.Sequential(nn.Conv3d(1, self.feature, kernel_size=(7, 7, 7), stride=(2, 2, 2),padding=(3, 3, 3), bias=False),
                                  nn.BatchNorm3d(self.feature),
                                  nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(BasicBlock, conv_makers[0], self.feature, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, conv_makers[1], self.feature*2, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, conv_makers[2], self.feature*4, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, conv_makers[3], self.feature*8, 2, stride=2)
        self.finalconv = nn.Sequential(nn.Conv3d(self.feature*8,self.out_dim,1,1,0),nn.BatchNorm3d(self.out_dim),nn.ReLU(inplace=True)) # 32+32 shape+location
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.de0 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                        nn.Conv3d(self.feature,self.feature,kernel_size=3,padding=1),
                        nn.BatchNorm3d(self.feature),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(self.feature,1,kernel_size=3,padding=1),
                        nn.Sigmoid()
        )
        self.de1 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                        nn.Conv3d(self.feature,self.feature,kernel_size=3,padding=1),
                        nn.BatchNorm3d(self.feature),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(self.feature,self.feature,kernel_size=3,padding=1),
                        nn.ReLU(inplace=True),
        )
        self.de2 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                        nn.Conv3d(self.feature*2,self.feature,kernel_size=3,padding=1),
                        nn.BatchNorm3d(self.feature),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(self.feature,self.feature,kernel_size=3,padding=1),
                        nn.BatchNorm3d(self.feature),
                        nn.ReLU(inplace=True)
        )
        self.de3 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                        nn.Conv3d(self.feature*2,self.feature*2,kernel_size=3,padding=1),
                        nn.BatchNorm3d(self.feature*2),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(self.feature*2,self.feature*2,kernel_size=3,padding=1),
                        nn.BatchNorm3d(self.feature*2),
                        nn.ReLU(inplace=True)
        )
        self.de4 = nn.Sequential(
                        nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True),
                        nn.Conv3d(self.out_dim,self.feature*2,kernel_size=3,padding=1),
                        nn.BatchNorm3d(self.feature*2),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(self.feature*2,self.feature*2,kernel_size=3,padding=1),
                        nn.BatchNorm3d(self.feature*2),
                        nn.ReLU(inplace=True)
        )

        self._initialize_weights()


    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self,x):
        x = self.stem(x) # 1*64^3 - 64*32^3
        x = self.layer1(x) # 64*32^3 - 64*32^3
        x2 = self.layer2(x) # 64*32^3 - 128*16^3
        x = self.layer3(x2) # 128*16^3 - 256*8^3
        x4 = self.layer4(x) # 256*8^3 - 512*4^3
        x4 = self.finalconv(x4)
        feat_total = self.avgpool(x4) # 512*1^3
        feat = feat_total[:,:64,:,:,:]
        x = self.de4(feat_total) # 512*4^3 - 256*4^3
        x = self.de3(x) # 256*4^3 - 128*8^3
        x = self.de2(x) # 128*8^3 - 64*16^3
        x = self.de1(x) # 64*16^3 - 64*32^3
        x = self.de0(x) # 64*32^3 - 1*64^3
        return x, feat


class R3D_18_Encoder_Split_ver2(nn.Module):
    def __init__(self, finetune=False):
        super(R3D_18_Encoder_Split_ver2, self).__init__()
        conv_makers = [Conv3DSimple] * 4

        self.out_dim = 64 + 64
        self.feature = 32
        self.inplanes = self.feature
        self.stem = nn.Sequential(nn.Conv3d(1, self.feature, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False),
                                  nn.BatchNorm3d(self.feature),
                                  nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(BasicBlock, conv_makers[0], self.feature, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, conv_makers[1], self.feature * 2, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, conv_makers[2], self.feature * 4, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, conv_makers[3], self.feature * 8, 2, stride=2)
        self.finalconv = nn.Sequential(nn.Conv3d(self.feature * 8, self.out_dim, 1, 1, 0), nn.BatchNorm3d(self.out_dim), nn.ReLU(inplace=True))  # 32+32 shape+location
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.flat = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(self.out_dim, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 1),
        )

        self.con_head = nn.Sequential(
            nn.Linear(self.out_dim, 128),
            nn.Linear(128, 64),
        )

        self.finetune = finetune
        if self.finetune:
            self.finetune_head = nn.Sequential(
                nn.Linear(64, 1),
            )

        self._initialize_weights()

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)  # 1*64^3 - 64*32^3
        x = self.layer1(x)  # 64*32^3 - 64*32^3
        x2 = self.layer2(x)  # 64*32^3 - 128*16^3
        x = self.layer3(x2)  # 128*16^3 - 256*8^3
        x4 = self.layer4(x)  # 256*8^3 - 512*4^3
        x4 = self.finalconv(x4)
        pool = self.avgpool(x4)  # 512*1^3
        flat = self.flat(pool)
        logits = self.classifier(flat)
        embedding_feature = self.con_head(flat)
        if self.finetune:
            embedding_feature = embedding_feature.detach()
            finetune_logits = self.finetune_head(embedding_feature)
            return logits, embedding_feature, finetune_logits
        return logits, embedding_feature


class R3D_18_Encoder_Split_ver2_finetune(nn.Module):
    def __init__(self, model_ckpt=None, linear=1, frozen=True, attach_clinic=False, attach_contrastive=False):
        super(R3D_18_Encoder_Split_ver2_finetune, self).__init__()
        self.linear = linear
        self.attach_clinic = attach_clinic
        if linear > 1:
            if attach_clinic:
                self.linears = nn.Sequential(
                    nn.Linear(64 + 7, 64),
                    nn.Linear(64, 64),
                )
            else:
                self.linears = nn.Sequential(
                    nn.Linear(64, 64),
                    nn.Linear(64, 64),
                )

        self.finetune_head = nn.Linear(64, 1)
        nn.init.normal_(self.finetune_head.weight, 0, 0.01)
        nn.init.constant_(self.finetune_head.bias, 0)

        self.net = R3D_18_Encoder_Split_ver2(finetune=False)
        checkpoints = torch.load(model_ckpt)
        self.net.load_state_dict(checkpoints['model'])

        self.frozen = frozen
        if frozen:
            self.net.eval()

        self.attach_con = attach_contrastive

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, clinic_feature=None):
        trans_logits, trans_embedding_feature = self.net(x)
        if self.frozen:
            trans_embedding_feature = trans_embedding_feature.detach()
        if self.linear > 1:
            if self.attach_clinic:
                clinic_feature = clinic_feature.detach()
                trans_embedding_feature_clinic = torch.cat((trans_embedding_feature, clinic_feature), dim=1)
                finetune_logits = self.linears(trans_embedding_feature_clinic)
                last_layer_feature = finetune_logits
            else:
                finetune_logits = self.linears(trans_embedding_feature)
            finetune_logits = self.finetune_head(finetune_logits)
        else:
            finetune_logits = self.finetune_head(trans_embedding_feature)
        if self.attach_con:
            return finetune_logits, trans_embedding_feature, last_layer_feature
        return finetune_logits


class R3D_18_Encoder_Split_ver4(nn.Module):
    def __init__(self):
        super(R3D_18_Encoder_Split_ver4, self).__init__()
        conv_makers = [Conv3DSimple] * 4

        self.out_dim = 64 + 64
        self.feature = 32
        self.inplanes = self.feature
        self.stem = nn.Sequential(nn.Conv3d(1, self.feature, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False),
                                  nn.BatchNorm3d(self.feature),
                                  nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(BasicBlock, conv_makers[0], self.feature, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, conv_makers[1], self.feature * 2, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, conv_makers[2], self.feature * 4, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, conv_makers[3], self.feature * 8, 2, stride=2)
        self.finalconv = nn.Sequential(nn.Conv3d(self.feature * 8, self.out_dim, 1, 1, 0), nn.BatchNorm3d(self.out_dim), nn.ReLU(inplace=True))  # 32+32 shape+location
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.flat = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(64, 1),
        )

        self.con_head = nn.Sequential(
            nn.Linear(self.out_dim, 128),
            nn.Linear(128, 64),
        )

        self._initialize_weights()

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)  # 1*64^3 - 64*32^3
        x = self.layer1(x)  # 64*32^3 - 64*32^3
        x2 = self.layer2(x)  # 64*32^3 - 128*16^3
        x = self.layer3(x2)  # 128*16^3 - 256*8^3
        x4 = self.layer4(x)  # 256*8^3 - 512*4^3
        x4 = self.finalconv(x4)
        pool = self.avgpool(x4)  # 512*1^3
        flat = self.flat(pool)
        head = self.con_head(flat)

        logits = self.classifier(head)
        return logits, head


class R3D_18(nn.Module):
    def __init__(self):
        super(R3D_18,self).__init__()
        conv_makers=[Conv3DSimple] * 4
        
        self.feature = 32
        self.inplanes = self.feature
        self.stem = nn.Sequential(nn.Conv3d(1, self.feature, kernel_size=(7, 7, 7), stride=(2, 2, 2),padding=(3, 3, 3), bias=False),
                                  nn.BatchNorm3d(self.feature),
                                  nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(BasicBlock, conv_makers[0], self.feature, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, conv_makers[1], self.feature*2, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, conv_makers[2], self.feature*4, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, conv_makers[3], self.feature*8, 4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.final = nn.Sequential(nn.Flatten(start_dim=1),nn.Linear(self.feature*8,64),nn.ReLU(inplace=True),nn.Dropout(0.5),nn.Linear(64,1),nn.Sigmoid())
        

        self._initialize_weights()


    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self,x):
        x = self.stem(x) # 1*64^3 - 64*32^3
        x = self.layer1(x) # 64*32^3 - 64*32^3
        x = self.layer2(x) # 64*32^3 - 128*16^3
        x = self.layer3(x) # 128*16^3 - 256*8^3
        x = self.layer4(x) # 256*8^3 - 512*4^3
        x = self.avgpool(x) # 512*1^3
        x = self.final(x)
        return x

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):

        super(Bottleneck, self).__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        # Second kernel
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Conv3DSimple(nn.Conv3d):
    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        super(Conv3DSimple, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)



