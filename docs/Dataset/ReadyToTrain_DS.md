## Brief description of the submodule

Here all of the code used to pass from the datasets downloaded to ready to use data for training is described.

## calculate_percentiles()

Function to calculate 0.01 and 0.99 percentiles of the bands of planet images. These values will be later used for normalizing the dataset.

### Params

* **img_folder:** (list) The name of the folder with the images.
* **samples:** (integer) The number of images to take to calculate these percentiles, for computing reasons not all images are considered.

### Outputs

* **vals:** (numpy.ndarray) The mean 1% and 99% quantiles for the images analysed.

### Dependencies used

```python
import os
import random
import numpy as np
import rioxarray
```

### Source code
```python
def calculate_percentiles(img_folder, samples = 400):
    """
        Function to calculate 0.01 and 0.99 percentiles of the bands of planet images. These values will be later used for normalizing the dataset.

        Inputs:
            - img_folder: The name of the folder with the images.
            - samples: The number of images to take to calculate these percentiles, for computing reasons not all images are considered.
        Output:
            - vals: The mean 1% and 99% quantiles for the images analysed.
    """
    imgs = [fn for fn in os.listdir(img_folder) if 'StudyArea' in fn]

    random.seed(8)
    img_sample = random.sample(imgs, samples)
    quantiles = np.zeros((2,4))
    
    for i in img_sample:
        quantiles += rioxarray.open_rasterio(img_folder + "\\" + i).quantile((0.01, 0.99), dim = ('x','y')).values
    
    vals = quantiles/len(img_sample)
    
    return vals
```

## get_DataLoaders()

Function to get the training, validation and test torch.DataLoader or torch.Dataset for a specific dataset. This function gets the images from the [Img_Dataset class](./ReadyToTrain_DS#img_dataset).
### Params

- **dir:** (str) Directory with the name of the data to be used.
- **batch_size:** (int) Size of the batches used for training.
- **transform:** (torchvision.transforms.V2.Compose) torch composition of transforms used for image augmentation.
- **normaliztion:** (str) Type of normalization used. (Should be 'Linear_1_99')
- **VI:** (boolean) Boolean indicating if NDVI and NDWI are also used in training.
- **split_size:** (float) Float between 0 and 1 indicating the fraction of dataset to be used (Especifically useful for HP tuning)
- **only_get_DS:** (boolean) Boolean for only getting datasets instead of dataloaders.
- **train_split_size:** (float) fraction of train split to be loaded. (number between 0 and 1)
- **val_split_size:** (float) fraction of validation and test split to be loaded. (number between 0 and 1)
  
### Outputs

Can be either the data loaders:

- **train_loader:** Training torch data loader
- **val_loader:** Validation torch data loader
- **test_loader:** Test torch data loader

or the datasets:

- **train_DS:** Training torch data set.
- **val_DS:** Validation torch data loader
- **test_DS:** Test torch data loader

### Dependencies used

```python
import torch
from torch.utils.data import random_split
```

### Source code

```python
def get_DataLoaders(dir, batch_size, transform, normalization, VI, only_get_DS = False, train_split_size = None, val_split_size = None):
    """
        Function to get the training, validation and test data loader for a specific dataset.

        Inputs:
            - dir: Directory with the name of the data to be used.
            - batch_size: Size of the batches used for training.
            - transform: torch composition of transforms used for image augmentation.
            - normaliztion: Type of normalization used.
            - VI: Boolean indicating if NDVI and NDWI are also used in training.
            - split_size: Float between 0 and 1 indicating the fraction of dataset to be used (Especifically useful for HP tuning)
            - only_get_DS: Boolean for only getting datasets instead of dataloaders.
        Output:
            - train_loader: Training torch data loader
            - val_loader: Validation torch data loader
            - test_loader: Test torch data loader
    """
    
    train_DS = Img_Dataset(dir, transform, norm = normalization, VI=VI)
    val_DS = Img_Dataset(dir, split = 'Validation', norm = normalization, VI=VI)
    test_DS = Img_Dataset(dir, split = 'Test', norm = normalization, VI=VI)

    if train_split_size != None:
        if val_split_size == None:
            val_split_size = train_split_size
            
        train_DS, l = random_split(train_DS, [train_split_size, 1-train_split_size], generator=torch.Generator().manual_seed(8))
        val_DS, l = random_split(val_DS, [val_split_size, 1-val_split_size], generator=torch.Generator().manual_seed(8))
        test_DS, l = random_split(test_DS, [val_split_size, 1-val_split_size], generator=torch.Generator().manual_seed(8))
        
    train_loader = torch.utils.data.DataLoader(dataset=train_DS, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_DS, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_DS, batch_size=batch_size, shuffle=False)
    
    if only_get_DS:
        return train_DS, val_DS, test_DS
    else:
        return train_loader, val_loader, test_loader
```

## get_LOVE_DataLoaders()

Function to get the loaders for LoveDA dataset, which was retrieved using [torchgeo](https://torchgeo.readthedocs.io/en/stable/api/datasets.html#loveda).

#### Size of the dataset:

|**Domain**|**Train**|**Validation**|**Test**|
|-|-|-|-|
|**Urban**|1,155|677|820|
|**Rural**|1,366|992|976|

### Params

- **domain:** List with the scene parameter for the LoveDa dataset. It can include 'rural' and/or 'urban'.
- **batch_size:** Number of images per batch.
- **transforms:** Image augmentations that will be considered.
- **only_get_DS:** Boolean for only getting datasets instead of dataloaders.
- **train_split_size:** Amount of images from training split that will be considered. (Float between 0 and 1)
- **val_split_size:** Amount of images from validation and test split that will be considered. (Float between 0 and 1)

### Outputs
- **train_loader:** Training torch LoveDA data loader
- **val_loader:** Validation torch LoveDA data loader
- **test_loader:** Test torch LoveDA data loader
  
### Dependencies used

```python
import torch
from torchgeo.datasets import LoveDA
from torch.utils.data import random_split
```
  
### Source code

```python
def get_LOVE_DataLoaders(domain = ['urban', 'rural'], batch_size = 4, transforms = None, only_get_DS = False, train_split_size = None, val_split_size = None):
    """
        Function to get the loaders for LoveDA dataset.

        Inputs:
            - domain: List with the scene parameter for the LoveDa dataset. It can include 'rural' and/or 'urban'.
            - batch_size: Number of images per batch.
            - transforms: Image augmentations that will be considered.
            - train_split_size: Amount of images from training split that will be considered. (Float between 0 and 1)
            - val_split_size: Amount of images from validation split that will be considered. (Float between 0 and 1)
            - only_get_DS: Boolean for only getting datasets instead of dataloaders.

        Output:
            - train_loader: Training torch LoveDA data loader
            - val_loader: Validation torch LoveDA data loader
            - test_loader: Test torch LoveDA data loader
    """
    if transforms != None:
        train_DS = LoveDA('LoveDA', split = 'train', scene = domain, download = True, transforms = transforms)
    else:
        train_DS = LoveDA('LoveDA', split = 'train', scene = domain, download = True)
        
    test_DS = LoveDA('LoveDA', split = 'test', scene = domain, download = True, transforms = transforms)
    val_DS = LoveDA('LoveDA', split = 'val', scene = domain, download = True, transforms = transforms)

    if train_split_size != None:
        if val_split_size == None:
            val_split_size = train_split_size
        train_DS, l = random_split(train_DS, [train_split_size, 1-train_split_size], generator=torch.Generator().manual_seed(8))
        val_DS, l = random_split(val_DS, [val_split_size, 1-val_split_size], generator=torch.Generator().manual_seed(8))
        test_DS, l = random_split(test_DS, [val_split_size, 1-val_split_size], generator=torch.Generator().manual_seed(8))
    
    train_loader = torch.utils.data.DataLoader(dataset=train_DS, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_DS, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_DS, batch_size=batch_size, shuffle=False)

    if only_get_DS:
        return train_DS, val_DS, test_DS
    else:
        return train_loader, val_loader, test_loader
```

## Img_Dataset

Class to manage the **Cashew dataset**. The Cashew dataset consists of 256x256 [Planet NICFI](https://www.planet.com/nicfi/?gad_source=1&gclid=CjwKCAiAk9itBhASEiwA1my_67JOFQ8L4DPicJ47w-b_bGBjLBM1SymMjL91UsJVmB5jSRwKsoedZxoCb2sQAvD_BwE) images with 4 bands(B, G, R, NIR).

#### Size of the dataset:

|**fold**|**Domain**|**Train**|**Validation**|**Test**|
|-|-|-|-|-|
|**1**|**IvoryCoast**|8225|411|38|
|**1**|**Tanzania**|1021|57|31|
|**2**|**IvoryCoast**|7770|214|337|
|**2**|**Tanzania**|1142|49|29|
|**3**|**IvoryCoast**|9267|466|120|
|**3**|**Tanzania**|1142|40|28|

#### Normalization

The normalization of the images in the dataset was performed using a linear normalization using the values of the percentiles of 1 and 99 percent. A nice explanation of image normalization can be found on this [medium post](https://medium.com/sentinel-hub/how-to-normalize-satellite-images-for-deep-learning-d5b668c885af). 

The equation used for the normalization is presented below:

$$
IMG_{normalized} = \frac{IMG - Perc_{1\%}}{Perc_{99\%} - Perc_{1\%}}
$$
#### Vegetation indices

Two additional channels can be added to the tensor which are the Normalized Difference Vegetation Index **(NDVI)** 

$$
NDVI = \frac{NIR - R}{NIR + R}
$$

and the Normalized Difference Water Index **(NDWI)**.

$$ 
NDWI = \frac{G - NIR}{G+NIR}
$$
### Attributes

- **self.img_folder** (str) Name of the folder in which the images are stored.
- **self.transform** (torchvision.transforms.V2.Compose) torch composition of transforms used for image augmentation.
- **self.split** (str) Split of the dataset to be retrieved. Can be Train, Validation or Test.
- **self.norm** (str) Type of normalization used. Only 'Linear_1_99' is allowed right now.
- **self.VI** (boolean) Boolean indicating if NDVI and NDWI are also used in training.

### Methods

#### \_\_len\_\_()

Method to calculate the number of images in the dataset.  

#### plot_imgs()

Method to plot a specific image of the dataset. Receives **idx** as the index of the image and **VIs** as a boolean to decide to plot or not the vegetation indices of the images.

#### \_\_getitem\_\_()

Method to get the tensors (image and ground truth) for a specific index (idx).

### Source code
```python
class Img_Dataset(Dataset):
    """
        Class to manage the cashew dataset.
    """
    def __init__(self, img_folder, transform = None, split = 'Train', norm = 'Linear_1_99', VI = True, recalculate_perc = False):
        self.img_folder = img_folder
        self.transform = transform
        self.split = split
        self.norm = norm
        self.VI = VI

        # Depending of the domain the images will have different attributes (country and quantiles)
        if 'Tanzania'  in self.img_folder:
            self.country = 'Tanzania'
            
            if recalculate_perc:
                self.quant_TNZ = calculate_percentiles(img_folder)
            else:
                self.quant_TNZ = quant_TNZ
        else:
            self.country = 'IvoryCoast'
            
            if recalculate_perc:
                self.quant_CIV = calculate_percentiles(img_folder)
            else:
                self.quant_CIV = quant_CIV

    def __len__(self):
        """
            Method to calculate the number of images in the dataset.    
        """
        return sum([self.split in i for i in os.listdir(self.img_folder)])//2

    def plot_imgs(self, idx, VIs = False):
        """
            Method to plot a specific image of the dataset.
            
            Input:
                - self: The dataset class and its attributes.
                - idx: index of the image that will be plotted.
                - VIs: Boolean describing if vegetation indices should be plotted
        """

        im, g = self.__getitem__(idx)

        if VIs:
            fig, ax = plt.subplots(2,2,figsize = (12,12))

            ax[0,0].imshow(im[[2,1,0],:,:].permute(1,2,0))
            ax[0,0].set_title('Planet image')
            ax[0,1].imshow(g[0,:,:])
            ax[0,1].set_title('Cashew crops GT')

            VIs = im[4:6]

            g1=ax[1,0].imshow(VIs[0], cmap = plt.cm.get_cmap('RdYlGn', 5), vmin = 0, vmax = 1)
            ax[1,0].set_title('NDVI')
            fig.colorbar(g1)
            g2=ax[1,1].imshow(VIs[1], cmap = plt.cm.get_cmap('Blues_r', 5), vmin = 0, vmax = 1)
            ax[1,1].set_title('NDWI')
            fig.colorbar(g2)

        else:
            fig, ax = plt.subplots(1,2,figsize = (12,6))

            ax[0].imshow(im[[2,1,0],:,:].permute(1,2,0))
            ax[0].set_title('Planet image')
            ax[1].imshow(g[0,:,:])
            ax[1].set_title('Cashew crops GT')


    def __getitem__(self, idx):
        """
            Method to get the tensors (image and ground truth) for a specific image.
        """
    
        conversion = T.ToTensor()

        img = io.imread(fname = self.img_folder + '/Cropped' + self.country + self.split + 'StudyArea_{:05d}'.format(idx) + '.tif').astype(np.float32)

        if self.VI:
            if self.norm == 'Linear_1_99':
                ndvi = (img[:,:,3] - img[:,:,2])/(img[:,:,3] + img[:,:,2]) 
                ndwi = (img[:,:,1] - img[:,:,3])/(img[:,:,3] + img[:,:,1])

        if self.norm == 'Linear_1_99':
            for i in range(img.shape[-1]):
                if 'Tanz' in self.img_folder:
                    img[:,:,i] = (img[:,:,i] - self.quant_TNZ[0,i])/(self.quant_TNZ[1,i] - self.quant_TNZ[0,i])
                elif 'Ivor' in self.img_folder:
                    img[:,:,i] = (img[:,:,i] - self.quant_CIV[0,i])/(self.quant_CIV[1,i] - self.quant_CIV[0,i])

        if self.VI:
            ndvi = np.expand_dims(ndvi, axis = 2)
            ndwi = np.expand_dims(ndwi, axis = 2)
            img = np.concatenate((img, ndvi, ndwi), axis = 2)

        img = conversion(img).float()

        img = torchvision.tv_tensors.Image(img)

        GT = io.imread(fname = self.img_folder + '/Cropped' + self.country + self.split + 'GT_{:05d}'.format(idx) + '.tif').astype(np.float32)

        GT = torch.flip(conversion(GT), dims = (1,))

        GT = torchvision.tv_tensors.Image(GT)

        if self.transform != None:
            GT, img = self.transform(GT, img)

        return img, GT
```