## Brief description of the submodule

Here all of the code used to pass from the datasets downloaded to ready to use data for training is described.

## calculate_percentiles()

Function to calculate 0.01 and 0.99 percentiles of the bands of planet images. These values will be later used for normalizing the dataset.

### Params

* **img_folder:** *(list)* The name of the folder with the images.
* **samples:** *(integer)* The number of images to take to calculate these percentiles, for computing reasons not all images are considered.

### Outputs

* **vals:** *(numpy.ndarray)* The mean 1% and 99% quantiles for the images analysed.

### Dependencies used

* os
* random
* numpy
* rioxarray

### Source code
```
    imgs = [fn for fn in os.listdir(img_folder) if 'StudyArea' in fn]

    random.seed(8)
    img_sample = random.sample(imgs, samples)
    quantiles = np.zeros((2,4))
    
    for i in img_sample:
        quantiles += rioxarray.open_rasterio(img_folder + "\\" + i).quantile((0.01, 0.99), dim = ('x','y')).values
    
    vals = quantiles/len(img_sample)
    
    return vals
```

## get_LOVE_DataLoaders()

Function to get the loaders for LoveDA dataset.

Size of the dataset:

|**Domain**|**Train**|**Validation**|**Test**|
|-|-|-|-|
|**Urban**|1,155|677|820|
|**Rural**|1,366|992|976|

### Params

### Outputs

### Dependencies used

* torchgeo.datasets -> LoveDA
* torch
  
### Source code

```python
def get_DataLoaders(dir, batch_size, transform, normalization, VI, only_get_DS = False, train_split_size = None, val_split_size = None):
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
        
    test_DS = LoveDA('LoveDA', split = 'test', scene = domain, download = True)
    val_DS = LoveDA('LoveDA', split = 'val', scene = domain, download = True)

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