## Brief description of the submodule

This submodule uses the building blocks from [Buildingblocks](./buildingblocks) to first build major blocks which consist of:

- **FE:** The feature extractor
- **C:** The classifier head of the [U-Net](#unet) and
- **D:** The discriminator head of the [U-Net + DANN](#unetdann).

And secondly build the models used for segmentation [U-Net](#unet) and domain adaptation purposes ([U-Net + DANN](#unetdann)).

## Major blocks

==ADD IMAGE OF UNET and UNET DANN dividing blocks==
### FE

Class containing the layers considered for the extraction of features from the 2D images. 

This feature extractor can change its size depending on the layer in which it is decided to divide the U-Net into feature extractor and classifier. 

The division of the U-Net was inspired by the implementation done by [brio et al. (2019)](link) for domain adaptation in 3D medical imagery.

#### Attributes

- **self.n_channels:** (int) Number of channels \[bands\] on the images that will enter the network.
- **self.starter:** (int) Number of feature maps obtained from the first double convolution performed. The number of feature maps obtained for every double convolution will be then calculated multiplying this value by $2ˆn$, where $n$ is the level in the U_net, starting from 0 in the upmost layer and 2 in the lowest layer.
- **self.up_layer:** (int : \[0-4\]) Number indicating the layer in which the network is divided into Feature Extractor and Classifier.
- **self.bilinear:** (Boolean) Boolean indicating the method for upsampling in the expanding path. ==Default== is True.
- **self.attention:** (Boolean) Boolean to indicate if attention gates will be added on the upsampling step. ==Default== is False.
- **self.resunet:** (Boolean) Boolean used to indicate if the double convolution will have a residual connection or not.==Default== is False.

#### Methods

- **DownSteps:** Function to get the resulting feature maps of each of the steps performed on the contracting path of the U-Net.
- **forward:** Function to perform the forward calculation of the features extracted.

#### Source code

```python
class FE(nn.Module):
    """
      Class for the creation of the feature extractor.
    """
    def __init__(self, n_channels, starter, up_layer, bilinear = True, attention = False, resunet = False):
        super(FE, self).__init__()

        self.n_channels = n_channels
        self.starter = starter
        self.bilinear = bilinear 
        self.up_layer = up_layer
        self.attention = attention
        self.resunet = resunet

        # Layers related to segmentation task
        self.inc = (DoubleConv(self.n_channels, self.starter, resunet = self.resunet))
        self.down1 = (Down(self.starter, self.starter*(2**1), resunet = self.resunet))
        self.down2 = (Down(self.starter*(2**1), self.starter*(2**2), resunet = self.resunet))
        self.down3 = (Down(self.starter*(2**2), self.starter*(2**3), resunet = self.resunet))
    
        factor = 2 if bilinear else 1
        
        self.down4 = (Down(self.starter*(2**3), self.starter*(2**4) // factor, resunet = self.resunet))
        
        if self.up_layer >= 1:
            self.up1 = (Up(self.starter*(2**4), self.starter*(2**3) // factor, bilinear, attention, resunet = self.resunet))
        if self.up_layer >= 2:
            self.up2 = (Up(self.starter*(2**3), self.starter*(2**2) // factor, bilinear, attention, resunet = self.resunet))
        if self.up_layer >= 3:
            self.up3 = (Up(self.starter*(2**2), self.starter*(2**1) // factor, bilinear, attention, resunet = self.resunet))
        if self.up_layer >= 4:
            self.up4 = (Up(self.starter*(2**1), self.starter, bilinear, attention, resunet = self.resunet))

    def DownSteps(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return x1, x2, x3, x4, x5

    def forward(self, x):

        # Downsample steps
        x1, x2, x3, x4, x5 = self.DownSteps(x)

        # Upsample steps
        if self.up_layer == 0:
            x = x5
        if self.up_layer >= 1:
            x = self.up1(x5, x4)
        if self.up_layer >= 2:
            x = self.up2(x, x3)
        if self.up_layer >= 3:
            x = self.up3(x, x2)
        if self.up_layer >= 4:
            x = self.up4(x, x1)

        return x
``` 

### C

Class containing the layers considered for the segmentation of the 2D images using the features extracted from [FE](##FE)

This classifier can change its size depending on the layer in which it is decided to divide the U-Net into feature extractor and classifier. 

The division of the U-Net was inspired by the implementation done by [brio et al. (2019)](link) for domain adaptation in 3D medical imagery.

#### Attributes

==NEED TO REMOVE N_CHANNELS AS ATTRIBUTE ==


- **self.starter:** (int) Number of feature maps obtained from the first double convolution performed. The number of feature maps obtained for every double convolution will be then calculated multiplying this value by $2ˆn$, where $n$ is the level in the U_net, starting from 0 in the upmost layer and 2 in the lowest layer.
- **self.up_layer:** (int : \[0-4\]) Number indicating the layer in which the network is divided into Feature Extractor and Classifier.
- **self.bilinear:** (Boolean) Boolean indicating the method for upsampling in the expanding path. ==Default== is True.
- **self.n_classes:** (int) Number of classes in which the images will be classified. ==Default== is 2.
- **self.attention:** (Boolean) Boolean to indicate if attention gates will be added on the upsampling step. ==Default== is False.
- **self.resunet:** (Boolean) Boolean used to indicate if the double convolution will have a residual connection or not.==Default== is False.

#### Methods

- **forward:** Function to perform the forward calculation of the logits. It takes as an input both the features extracted in [FE](###FE) and the feature maps extracted from the downsampling steps which can be obtained with the [DownSteps](####methods)  method.

#### Source code

```python
class C(nn.Module):
    def __init__(self, n_channels, starter, up_layer, bilinear = True, n_classes = 2, attention = False, resunet = False):
        super(C, self).__init__()

        self.n_channels = n_channels
        self.bilinear = bilinear
        self.starter = starter
        self.up_layer = up_layer
        self.n_classes = n_classes
        self.attention = attention
        self.resunet = resunet

        factor = 2 if bilinear else 1

        if self.up_layer == 0:
            self.up1 = (Up(self.starter*(2**4), self.starter*(2**3) // factor, bilinear, attention, resunet = self.resunet))
            self.up2 = (Up(self.starter*(2**3), self.starter*(2**2) // factor, bilinear, attention, resunet = self.resunet))
            self.up3 = (Up(self.starter*(2**2), self.starter*(2**1) // factor, bilinear, attention, resunet = self.resunet))
            self.up4 = (Up(self.starter*(2**1), self.starter, bilinear, attention, self.resunet))
            self.outc = (OutConv(self.starter, n_classes))
        elif self.up_layer == 1:
            self.up2 = (Up(self.starter*(2**3), self.starter*(2**2) // factor, bilinear, attention, resunet = self.resunet))
            self.up3 = (Up(self.starter*(2**2), self.starter*(2**1) // factor, bilinear, attention, resunet = self.resunet))
            self.up4 = (Up(self.starter*(2**1), self.starter, bilinear, attention, resunet = self.resunet))
            self.outc = (OutConv(self.starter, n_classes))
        elif self.up_layer == 2:
            self.up3 = (Up(self.starter*(2**2), self.starter*(2**1) // factor, bilinear, attention, resunet = self.resunet))
            self.up4 = (Up(self.starter*(2**1), self.starter, bilinear, attention, resunet = self.resunet))
            self.outc = (OutConv(self.starter, n_classes))
        elif self.up_layer == 3:
            self.up4 = (Up(self.starter*(2**1), self.starter, bilinear, attention, resunet = self.resunet))
            self.outc = (OutConv(self.starter, n_classes))
        elif self.up_layer == 4:
            self.outc = (OutConv(self.starter, n_classes))


    def forward(self, x, dw):
        # Downsample steps
        x1, x2, x3, x4, x5 = dw

        # Upsampling steps
        if self.up_layer == 0:
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
        elif self.up_layer == 1:
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
        elif self.up_layer == 2:
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
        elif self.up_layer == 3:
            x = self.up4(x, x1)
            logits = self.outc(x)
        elif self.up_layer == 4:
            logits = self.outc(x)

        return logits
```

### D

Class containing the layers used to discriminate between the source and target domain using the features extracted in [FE](###FE). 

The discriminator first downsamples the features using the same class [Down](./BuildingBlocks#Down) used in the contracting path of the U-Net. Then, a [fully connected layer](./BuildingBlocks#outdisc) is used to discriminate between source and target domain.

#### Attributes

- **self.initial_features:** (int) Number of features that will go into the final fully connected layer.
- **self.starter:** (int) Number of feature maps obtained from the first double convolution performed. The number of feature maps obtained for every double convolution will be then calculated multiplying this value by $2ˆn$, where $n$ is the level in the U_net, starting from 0 in the upmost layer and 2 in the lowest layer.
- **self.up_layer:** (int : \[0-4\]) Number indicating the layer in which the network is divided into Feature Extractor and Classifier.
- **self.bilinear:** (Boolean) Boolean indicating the method for upsampling in the expanding path. ==Default== is True.
- **self.resunet:** (Boolean) Boolean used to indicate if the double convolution will have a residual connection or not.==Default== is False.
- **self.grad_rev_w:** (float) Number with the learning weight of the discriminator head. ==Default== is 1.

#### Methods

- **forward:** Function to perform the forward calculation of the logits. This function takes the features extracted from [FE](#FE) and the value of grad_rev_w, which is the same as constant $\lambda$ from [GradReverse](./BuildingBlocks#gradreverse).

#### Source code

```python
class D(nn.Module):
    def __init__(self, initial_features, bilinear=True, starter = 8, up_layer = 3, resunet = False, grad_rev_w = 1):
        super(D, self).__init__()

        self.initial_features = initial_features
        self.bilinear = bilinear
        self.starter = starter
        self.up_layer = up_layer
        self.resunet = resunet
        self.grad_rev_w = grad_rev_w

        factor = 2 if bilinear else 1

        self.revgrad = GradReverse.grad_reverse
        
        self.outd = (OutDisc(self.initial_features, 256))

        if self.up_layer > 0:
            self.down4_D = (Down(self.starter*(2**3)//factor, self.starter*(2**4)//factor, resunet = self.resunet))
        if self.up_layer > 1:
            self.down3_D = (Down(self.starter*(2**2)//factor, self.starter*(2**3)//factor, resunet = self.resunet))
        if self.up_layer > 2:
            self.down2_D = (Down(self.starter*(2**1)//factor, self.starter*(2**2)//factor, resunet = self.resunet))
        if self.up_layer > 3:
            self.down1_D = (Down(self.starter, self.starter*2//factor, resunet = self.resunet))
            
    def forward(self, x, grad_rev_w):

        x = self.revgrad(x, grad_rev_w)

        if self.up_layer == 1:
            x = self.down4_D(x)
        if self.up_layer == 2:
            x = self.down3_D(x)
            x = self.down4_D(x)
        if self.up_layer == 3:
            x = self.down2_D(x)
            x = self.down3_D(x)
            x = self.down4_D(x)
        if self.up_layer == 4:
            x = self.down1_D(x)
            x = self.down2_D(x)
            x = self.down3_D(x)
            x = self.down4_D(x)

        x = self.outd(x)

        return x
```

## Networks implemented
### UNet

Class containing the layers used to create the custom UNet used to segment 2D images. An overview of the network can be seen [here](#overview). 

#### Overview
==IMAGE==

#### Attributes

- **self.n_channels:** (int) Number of channels \[bands\] on the images that will enter the network.
- **self.n_classes:** (int) Number of classes in which the images will be classified.
- **self.bilinear:** (Boolean) Boolean indicating the method for upsampling in the expanding path. ==Default== is True.
- **self.starter:** (int) Number of feature maps obtained from the first double convolution performed. The number of feature maps obtained for every double convolution will be then calculated multiplying this value by $2ˆn$, where $n$ is the level in the U_net, starting from 0 in the upmost layer and 2 in the lowest layer. ==Default== is 8.
- **self.up_layer:** (int : \[0-4\]) Number indicating the layer in which the network is divided into Feature Extractor and Classifier. ==Default== is 3.
- **self.attention:** (Boolean) Boolean to indicate if attention gates will be added on the upsampling step. ==Default== is False.
- **self.resunet:** (Boolean) Boolean used to indicate if the double convolution will have a residual connection or not.==Default== is False.

#### Methods

- **forward:** Function to perform the forward calculation of the logits per class using images as an input.
- **init_weights:** Function to initialize the weights using the xavier normal method.

#### Source code

```python
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, starter = 8, up_layer = 3, attention = False, resunet = False):

        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.starter = starter
        self.up_layer = up_layer
        self.attention = attention
        self.resunet = resunet

        self.FE = (FE(self.n_channels, self.starter, self.up_layer, self.bilinear, self.attention, resunet = self.resunet))
        self.C = (C(self.n_channels, self.starter, self.up_layer, self.bilinear, self.n_classes, self.attention, resunet = self.resunet))

        self.apply(self._init_weights)

    def forward(self, x):

        features = self.FE(x) # Feature extractor
        down_st = self.FE.DownSteps(x) # Get channels that will be concatenated from downward steps

        logits = self.C(features, down_st) # Classifier
        
        return logits


    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
```

### UNetDANN

#### Overview

![UnetDANN](/img/U_Net_DANN.png)

#### Attributes 

#### Methods

- **forward:** Function to perform the forward calculation of the logits per class and the logits per domain using images as an input. This networks has two heads, therefore it returns two results.
- **init_weights:** Function to initialize the weights using the xavier normal method.

#### Source code

```python
class UNetDANN(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, starter = 8, up_layer = 3, attention = False, resunet = False, DA = False, in_feat = None, grad_rev_w = 1):

        super(UNetDANN, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.starter = starter
        self.up_layer = up_layer
        self.attention = attention
        self.resunet = resunet
        self.in_feat = in_feat
        self.DA = DA

        self.FE = (FE(self.n_channels, self.starter, self.up_layer, self.bilinear, self.attention, resunet = self.resunet))
        self.C = (C(self.n_channels, self.starter, self.up_layer, self.bilinear, self.n_classes, self.attention, resunet = self.resunet))

        if DA:
            self.D = (D(initial_features=self.in_feat, bilinear = self.bilinear, starter = self.starter, up_layer = self.up_layer, resunet = self.resunet, grad_rev_w = grad_rev_w))

        self.apply(self._init_weights)

    def forward(self, x, grad_rev_w = 0):

        features = self.FE(x) # Feature extractor
        down_st = self.FE.DownSteps(x) # Get channels that will be concatenated from downward steps

        logits = self.C(features, down_st) # Classifier

        dom_preds = self.D(features, grad_rev_w)

        return logits, dom_preds


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
```