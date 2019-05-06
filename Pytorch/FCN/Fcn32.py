import torch
import torch.nn as nn
import numpy as np 

def get_upsampling_weight(in_channels,out_channels,kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""



class FCN32(nn.Module):

    def __init__(self, n_class= 2):#VALUE
        super(FCN32,self).__init__()

        #convolution 1

        self.conv1_1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)                # 1/2


        #convolution 2

        self.conv2_1 = nn.Conv2d(in_channels=65,out_channels=128,kernel_size=3,padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)                # 1/4


        #convolution 3

        self.conv3_1 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)                # 1/8

        #convolution 4

        self.conv4_1 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2,stride=3,ceil_mode=True)                # 1/16


        #convolution 5
        self.conv5_1 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)                # 1/32


        #fc 6
        self.fc6 = nn.Conv2d(in_channels=512,out_channels=4096,kernel_size=7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        #fc 7
        self.fc7 = nn.Conv2d(in_channels=4096,out_channels=4096,kernel_size=7)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()



        self.score_fn = nn.Conv2d(in_channels=4096,out_channels=n_class,kernel_size=1)
        self.upscore = nn.ConvTranspose2d(n_class,n_class,64,stride=32,bias=False)

        self._initialize_weights()



#FIGURING OUT
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)
    


    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)

        h = self.upscore(h)
        h = h[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()

        return h



    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())





        



