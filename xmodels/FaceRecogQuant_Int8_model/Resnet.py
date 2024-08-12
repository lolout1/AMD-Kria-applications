# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class Resnet(torch.nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.module_0 = py_nndct.nn.Input() #Resnet::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Resnet::Resnet/Sequential/MyBlock[0]/Sequential[downsample]/Conv2d[0]/input.2
        self.module_2 = py_nndct.nn.ReLU(inplace=False) #Resnet::Resnet/Sequential/MyBlock[0]/Sequential[downsample]/ReLU[1]/input.3
        self.module_3 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Resnet::Resnet/Sequential/MyBlock[0]/Conv2d[conv1]/input.4
        self.module_5 = py_nndct.nn.ReLU(inplace=False) #Resnet::Resnet/Sequential/MyBlock[0]/ReLU[relu1]/input.6
        self.module_6 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Resnet::Resnet/Sequential/MyBlock[0]/Conv2d[conv2]/input.7
        self.module_8 = py_nndct.nn.Add() #Resnet::Resnet/Sequential/MyBlock[0]/input.8
        self.module_9 = py_nndct.nn.ReLU(inplace=False) #Resnet::Resnet/Sequential/MyBlock[0]/ReLU[relu2]/input.9
        self.module_10 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Resnet::Resnet/Sequential/MyBlock[0]/Sequential[downsample]/Conv2d[0]/input.10
        self.module_11 = py_nndct.nn.ReLU(inplace=False) #Resnet::Resnet/Sequential/MyBlock[0]/Sequential[downsample]/ReLU[1]/input.11
        self.module_12 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Resnet::Resnet/Sequential/MyBlock[0]/Conv2d[conv1]/input.12
        self.module_14 = py_nndct.nn.ReLU(inplace=False) #Resnet::Resnet/Sequential/MyBlock[0]/ReLU[relu1]/input.14
        self.module_15 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Resnet::Resnet/Sequential/MyBlock[0]/Conv2d[conv2]/input.15
        self.module_17 = py_nndct.nn.Add() #Resnet::Resnet/Sequential/MyBlock[0]/input.16
        self.module_18 = py_nndct.nn.ReLU(inplace=False) #Resnet::Resnet/Sequential/MyBlock[0]/ReLU[relu2]/input.17
        self.module_19 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Resnet::Resnet/Sequential/MyBlock[1]/Conv2d[conv1]/input.18
        self.module_21 = py_nndct.nn.ReLU(inplace=False) #Resnet::Resnet/Sequential/MyBlock[1]/ReLU[relu1]/input.20
        self.module_22 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Resnet::Resnet/Sequential/MyBlock[1]/Conv2d[conv2]/input.21
        self.module_24 = py_nndct.nn.Add() #Resnet::Resnet/Sequential/MyBlock[1]/input.22
        self.module_25 = py_nndct.nn.ReLU(inplace=False) #Resnet::Resnet/Sequential/MyBlock[1]/ReLU[relu2]/input.23
        self.module_26 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Resnet::Resnet/Sequential/MyBlock[0]/Sequential[downsample]/Conv2d[0]/input.24
        self.module_27 = py_nndct.nn.ReLU(inplace=False) #Resnet::Resnet/Sequential/MyBlock[0]/Sequential[downsample]/ReLU[1]/input.25
        self.module_28 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Resnet::Resnet/Sequential/MyBlock[0]/Conv2d[conv1]/input.26
        self.module_30 = py_nndct.nn.ReLU(inplace=False) #Resnet::Resnet/Sequential/MyBlock[0]/ReLU[relu1]/input.28
        self.module_31 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Resnet::Resnet/Sequential/MyBlock[0]/Conv2d[conv2]/input.29
        self.module_33 = py_nndct.nn.Add() #Resnet::Resnet/Sequential/MyBlock[0]/input.30
        self.module_34 = py_nndct.nn.ReLU(inplace=False) #Resnet::Resnet/Sequential/MyBlock[0]/ReLU[relu2]/input.31
        self.module_35 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Resnet::Resnet/Sequential/MyBlock[1]/Conv2d[conv1]/input.32
        self.module_37 = py_nndct.nn.ReLU(inplace=False) #Resnet::Resnet/Sequential/MyBlock[1]/ReLU[relu1]/input.34
        self.module_38 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Resnet::Resnet/Sequential/MyBlock[1]/Conv2d[conv2]/input.35
        self.module_40 = py_nndct.nn.Add() #Resnet::Resnet/Sequential/MyBlock[1]/input.36
        self.module_41 = py_nndct.nn.ReLU(inplace=False) #Resnet::Resnet/Sequential/MyBlock[1]/ReLU[relu2]/input.37
        self.module_42 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Resnet::Resnet/Sequential/MyBlock[2]/Conv2d[conv1]/input.38
        self.module_44 = py_nndct.nn.ReLU(inplace=False) #Resnet::Resnet/Sequential/MyBlock[2]/ReLU[relu1]/input.40
        self.module_45 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Resnet::Resnet/Sequential/MyBlock[2]/Conv2d[conv2]/input.41
        self.module_47 = py_nndct.nn.Add() #Resnet::Resnet/Sequential/MyBlock[2]/input.42
        self.module_48 = py_nndct.nn.ReLU(inplace=False) #Resnet::Resnet/Sequential/MyBlock[2]/ReLU[relu2]/input.43
        self.module_49 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Resnet::Resnet/Sequential/MyBlock[3]/Conv2d[conv1]/input.44
        self.module_51 = py_nndct.nn.ReLU(inplace=False) #Resnet::Resnet/Sequential/MyBlock[3]/ReLU[relu1]/input.46
        self.module_52 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Resnet::Resnet/Sequential/MyBlock[3]/Conv2d[conv2]/input.47
        self.module_54 = py_nndct.nn.Add() #Resnet::Resnet/Sequential/MyBlock[3]/input.48
        self.module_55 = py_nndct.nn.ReLU(inplace=False) #Resnet::Resnet/Sequential/MyBlock[3]/ReLU[relu2]/input.49
        self.module_56 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Resnet::Resnet/Sequential/MyBlock[0]/Sequential[downsample]/Conv2d[0]/input.50
        self.module_57 = py_nndct.nn.ReLU(inplace=False) #Resnet::Resnet/Sequential/MyBlock[0]/Sequential[downsample]/ReLU[1]/input.51
        self.module_58 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Resnet::Resnet/Sequential/MyBlock[0]/Conv2d[conv1]/input.52
        self.module_60 = py_nndct.nn.ReLU(inplace=False) #Resnet::Resnet/Sequential/MyBlock[0]/ReLU[relu1]/input.54
        self.module_61 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Resnet::Resnet/Sequential/MyBlock[0]/Conv2d[conv2]/input.55
        self.module_63 = py_nndct.nn.Add() #Resnet::Resnet/Sequential/MyBlock[0]/input.56
        self.module_64 = py_nndct.nn.ReLU(inplace=False) #Resnet::Resnet/Sequential/MyBlock[0]/ReLU[relu2]/input.57
        self.module_66 = py_nndct.nn.Module('shape') #Resnet::Resnet/443
        self.module_67 = py_nndct.nn.Module('reshape') #Resnet::Resnet/input
        self.module_68 = py_nndct.nn.Linear(in_features=21504, out_features=512, bias=False) #Resnet::Resnet/Linear/448

    def forward(self, *args):
        self.output_module_0 = self.module_0(input=args[0])
        self.output_module_1 = self.module_1(self.output_module_0)
        self.output_module_2 = self.module_2(self.output_module_1)
        self.output_module_3 = self.module_3(self.output_module_2)
        self.output_module_5 = self.module_5(self.output_module_3)
        self.output_module_6 = self.module_6(self.output_module_5)
        self.output_module_8 = self.module_8(alpha=1, other=self.output_module_2, input=self.output_module_6)
        self.output_module_9 = self.module_9(self.output_module_8)
        self.output_module_10 = self.module_10(self.output_module_9)
        self.output_module_11 = self.module_11(self.output_module_10)
        self.output_module_12 = self.module_12(self.output_module_11)
        self.output_module_14 = self.module_14(self.output_module_12)
        self.output_module_15 = self.module_15(self.output_module_14)
        self.output_module_17 = self.module_17(alpha=1, other=self.output_module_11, input=self.output_module_15)
        self.output_module_18 = self.module_18(self.output_module_17)
        self.output_module_19 = self.module_19(self.output_module_18)
        self.output_module_21 = self.module_21(self.output_module_19)
        self.output_module_22 = self.module_22(self.output_module_21)
        self.output_module_24 = self.module_24(alpha=1, other=self.output_module_18, input=self.output_module_22)
        self.output_module_25 = self.module_25(self.output_module_24)
        self.output_module_26 = self.module_26(self.output_module_25)
        self.output_module_27 = self.module_27(self.output_module_26)
        self.output_module_28 = self.module_28(self.output_module_27)
        self.output_module_30 = self.module_30(self.output_module_28)
        self.output_module_31 = self.module_31(self.output_module_30)
        self.output_module_33 = self.module_33(alpha=1, other=self.output_module_27, input=self.output_module_31)
        self.output_module_34 = self.module_34(self.output_module_33)
        self.output_module_35 = self.module_35(self.output_module_34)
        self.output_module_37 = self.module_37(self.output_module_35)
        self.output_module_38 = self.module_38(self.output_module_37)
        self.output_module_40 = self.module_40(alpha=1, other=self.output_module_34, input=self.output_module_38)
        self.output_module_41 = self.module_41(self.output_module_40)
        self.output_module_42 = self.module_42(self.output_module_41)
        self.output_module_44 = self.module_44(self.output_module_42)
        self.output_module_45 = self.module_45(self.output_module_44)
        self.output_module_47 = self.module_47(alpha=1, other=self.output_module_41, input=self.output_module_45)
        self.output_module_48 = self.module_48(self.output_module_47)
        self.output_module_49 = self.module_49(self.output_module_48)
        self.output_module_51 = self.module_51(self.output_module_49)
        self.output_module_52 = self.module_52(self.output_module_51)
        self.output_module_54 = self.module_54(alpha=1, other=self.output_module_48, input=self.output_module_52)
        self.output_module_55 = self.module_55(self.output_module_54)
        self.output_module_56 = self.module_56(self.output_module_55)
        self.output_module_57 = self.module_57(self.output_module_56)
        self.output_module_58 = self.module_58(self.output_module_57)
        self.output_module_60 = self.module_60(self.output_module_58)
        self.output_module_61 = self.module_61(self.output_module_60)
        self.output_module_63 = self.module_63(alpha=1, other=self.output_module_57, input=self.output_module_61)
        self.output_module_64 = self.module_64(self.output_module_63)
        self.output_module_66 = self.module_66(input=self.output_module_64, dim=0)
        self.output_module_67 = self.module_67(input=self.output_module_64, size=[self.output_module_66,-1])
        self.output_module_68 = self.module_68(self.output_module_67)
        return self.output_module_68
