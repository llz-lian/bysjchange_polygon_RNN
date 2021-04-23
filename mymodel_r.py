from pathlib import Path

import torch
import torch.nn.init
from torch.nn.modules import linear
from torch.nn.modules.activation import ReLU
import torch.utils.model_zoo as model_zoo
import wget
from torch import nn, relu, softmax

from utils.convlstm import ConvLSTM
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        """if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)"""

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)#244*244*3 => 112*112*64
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)#112*112*64=>56*56*64

        x1 = self.layer1(x)#56*56*64=>56*56*256    --->maxpolling(2,2)
        x2 = self.layer2(x1)#56*56*256=>28*28*512 ---->conv(512,256)
        x3 = self.layer3(x2)#28*28*512=>14*14*1024---->conv(1024,256)---->upscaling
        x4 = self.layer4(x3)#14*14*1024=>7*7*2048---->upscaling()*2---->conv(2048,256)
        """if self.include_top:
            x = self.avgpool(x4)
            x = torch.flatten(x, 1)
            x = self.fc(x)"""
        return [x1,x2,x3,x4]


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder_part1 =nn.Sequential( 
            nn.Linear(28*28+3,112*56),
            nn.ReLU()
        )
        self.decoder_part2 =nn.Sequential( 
            nn.Linear(112*56,112*112+3),
            nn.ReLU()
        ) 
        self.decoders1 = nn.Sequential(
            nn.Linear(28*28+3,56*56),
            nn.ReLU(),
            nn.Linear(56*56,112*56),
            
            )
        self.decoders2=nn.Sequential(
            nn.ReLU(),
            nn.Linear(112*56,112*78),
            nn.ReLU()
        )
        self.decoders_end = nn.Linear(112*78,112*112+3)
    def forward(self,x):
        output_part = self.decoder_part1(x)#112*56
        output = self.decoders1(x)#112*56
        output += output_part

        output_part = self.decoder_part2(output)
        output = self.decoders2(output)

        output += output_part
        output = self.decoders_end(output)
        return output





class PolygonNet(nn.Module):

    def __init__(self, load_vgg=True,num_class = 8):
        super(PolygonNet, self).__init__()

        def _make_basic(input_size, output_size, kernel_size, stride, padding):
            """

            :rtype: nn.Sequential
            """
            return nn.Sequential(
                nn.Conv2d(input_size, output_size, kernel_size, stride,
                          padding),
                nn.ReLU(),
                nn.BatchNorm2d(output_size)
            )


        self.res = ResNet(Bottleneck, [3, 4, 6, 3],  include_top=True,num_classes=num_class)

        self.convlayer1 = _make_basic(512, 256, 3, 1, 1)
        self.convlayer2 = _make_basic(1024, 256, 3, 1, 1)
        self.convlayer3 = _make_basic(2048, 256, 3, 1, 1)
        self.convlayer5 = _make_basic(1024, 128, 3, 1, 1)
        self.poollayer = nn.MaxPool2d(2, 2).cuda()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear').cuda()

        self.convlstm = ConvLSTM(input_size=(28, 28),
                                 input_dim=131,
                                 hidden_dim=[32, 8],
                                 kernel_size=(3, 3),
                                 num_layers=2,
                                 batch_first=True,
                                 bias=True,
                                 return_all_layers=True)
        self.lstmlayer = nn.LSTM(28 * 28 * 8 + (28 * 28 + 3) * 2, 28 * 28 * 2,
                                 batch_first=True)
        self.linear = nn.Linear(28 * 28 * 2, 28 * 28 + 3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        self.fc = nn.Linear(512 * 4, num_class)
        """
        self.decoder = nn.Sequential(
            nn.Linear(28*28+3,56*56),
            nn.Tanh(),
            nn.Linear(56*56,112*56),
            nn.Tanh(),
            nn.Linear(112*56,112*112+3),
        )
        """

        


        self.decoder = Decoder()
        self.init_weights(load_vgg=load_vgg)

    def init_weights(self, load_vgg=True):
        '''
        Initialize weights of PolygonNet
        :param load_vgg: bool
                    load pretrained vgg model or not
        '''

        for name, param in self.convlstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.lstmlayer.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 1.0)
            elif 'weight' in name:
                # nn.init.xavier_normal_(param)
                nn.init.orthogonal_(param)
        for name, param in self.named_parameters():
            if 'bias' in name and 'convlayer' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name and 'convlayer' in name and '0' in name:
                nn.init.xavier_normal_(param)
        """if load_vgg:
            if vgg_file.is_file():
                vgg16_dict = torch.load('vgg16_bn-6c64b313.pth')
            else:
                try:
                    wget.download(
                        'https://download.pytorch.org/models/vgg16_bn'
                        '-6c64b313.pth')
                    vgg16_dict = torch.load('vgg16_bn-6c64b313.pth')
                except:
                    vgg16_dict = torch.load(model_zoo.load_url(
                        'https://download.pytorch.org/models/vgg16_bn'
                        '-6c64b313.pth'))
            vgg_name = []
            for name in vgg16_dict:
                if 'feature' in name and 'running' not in name:
                    vgg_name.append(name)
            cnt = 0
            # print([x[0] for x in self.named_parameters()])
            for name, param in self.named_parameters():
                if 'model' in name:
                    param.data.copy_(vgg16_dict[vgg_name[cnt]])
                    cnt += 1"""

    def forward(self, input_data1, first, second, third):
        bs = second.shape[0]
        length_s = second.shape[1]
        
        output1,output2,output3,output4 = self.res(input_data1)
        classes = self.avgpool(output4)
        classes = torch.flatten(classes,1)
        classes = self.fc(classes)


        output11 = self.poollayer(output1)

        output22 = self.convlayer1(output2)

        output33 = self.convlayer2(output3)
        output33 = self.upsample(output33)

        output44 = self.convlayer3(output4)
        output44 = self.upsample(output44)
        output44 = self.upsample(output44)


        output = torch.cat([output11, output22, output33, output44], dim=1)
        output = self.convlayer5(output)
        """
        bb+classes
        
        """

        output = output.unsqueeze(1)#28*28*128=>28*1*28*128
        output = output.repeat(1, length_s, 1, 1, 1)#28*1*28*128 => 28*length_s*28*128
        padding_f = torch.zeros([bs, 1, 1, 28, 28]).cuda()

        input_f = first[:, :-3].view(-1, 1, 28, 28).unsqueeze(1).repeat(1,
                                                                        length_s - 1,
                                                                        1, 1,
                                                                        1)
        input_f = torch.cat([padding_f, input_f], dim=1)
        input_s = second[:, :, :-3].view(-1, length_s, 1, 28, 28)
        input_t = third[:, :, :-3].view(-1, length_s, 1, 28, 28)
        output = torch.cat([output, input_f, input_s, input_t], dim=2)

        output = self.convlstm(output)[0][-1]

        shape_o = output.shape
        output = output.contiguous().view(bs, length_s, -1)
        output = torch.cat([output, second, third], dim=2)
        output = self.lstmlayer(output)[0]
        output = output.contiguous().view(bs * length_s, -1)
        output = self.linear(output)
        output = self.decoder(output)

        output = output.contiguous().view(bs, length_s, -1)

        return output,classes
    def test(self, input_data1, len_s):
        bs = input_data1.shape[0]
        result = torch.zeros([bs, len_s]).cuda()

        output1,output2,output3,output4 = self.res(input_data1)
        output11 = self.poollayer(output1)

        output22 = self.convlayer1(output2)

        output33 = self.convlayer2(output3)
        output33 = self.upsample(output33)

        output44 = self.convlayer3(output4)
        output44 = self.upsample(output44)
        output44 = self.upsample(output44)
        classes = self.avgpool(output4)
        classes = torch.flatten(classes,1)
        classes = self.fc(classes)

        output = torch.cat([output11, output22, output33, output44], dim=1)


        feature = self.convlayer5(output)

        padding_f = torch.zeros([bs, 1, 1, 28, 28]).float().cuda()
        input_s = torch.zeros([bs, 1, 1, 28, 28]).float().cuda()
        input_t = torch.zeros([bs, 1, 1, 28, 28]).float().cuda()

        output = torch.cat([feature.unsqueeze(1), padding_f, input_s, input_t],
                           dim=2)

        output, hidden1 = self.convlstm(output)
        output = output[-1]
        output = output.contiguous().view(bs, 1, -1)
        second = torch.zeros([bs, 1, 28 * 28 + 3]).cuda()
        second[:, 0, 28 * 28 + 1] = 1
        third = torch.zeros([bs, 1, 28 * 28 + 3]).cuda()
        third[:, 0, 28 * 28 + 2] = 1
        output = torch.cat([output, second, third], dim=2)

        output, hidden2 = self.lstmlayer(output)
        output = output.contiguous().view(bs, -1)
        output = self.linear(output)
        outputs = self.decoder(output)

        outputs = outputs.contiguous().view(bs,1,-1)
        
        outputs = (outputs == outputs.max(dim=2, keepdim=True)[0]).float()
        output = output.contiguous().view(bs, 1, -1)
        output = (output == output.max(dim=2, keepdim=True)[0]).float()

        first = output
        result[:, 0] = (outputs.argmax(2))[:, 0]

        for i in range(len_s - 1):
            second = third
            third = output
            input_f = first[:, :, :-3].view(-1, 1, 1, 28, 28)
            input_s = second[:, :, :-3].view(-1, 1, 1, 28, 28)
            input_t = third[:, :, :-3].view(-1, 1, 1, 28, 28)
            input1 = torch.cat(
                [feature.unsqueeze(1), input_f, input_s, input_t], dim=2)
            output, hidden1 = self.convlstm(input1, hidden1)
            output = output[-1]
            output = output.contiguous().view(bs, 1, -1)
            output = torch.cat([output, second, third], dim=2)
            output, hidden2 = self.lstmlayer(output, hidden2)
            output = output.contiguous().view(bs, -1)
            output = self.linear(output)
            outputs = self.decoder(output)
            output = output.contiguous().view(bs, 1, -1)
            outputs = outputs.contiguous().view(bs,1,-1)
            output = (output == output.max(dim=2, keepdim=True)[0]).float()
            outputs = (outputs == outputs.max(dim=2, keepdim=True)[0]).float()
            result[:, i + 1] = (outputs.argmax(2))[:, 0]
        return result,classes
