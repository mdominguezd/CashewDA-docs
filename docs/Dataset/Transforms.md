# Transforms

## Brief description of the submodule

In this submodule you can find the functions used for augmneting the datasets during training.

## brightness()

Function to augment images. It increases or decreases the brightness of the image by a value of 0 to 10%.

### Params

- **i:** (torch.tensor) Tensor to be augmented.
- **increase:** (float) Percentage of increase or decrease of brightness in the images. Default is 0.1.
- **prob:** (float) Probability of changing the brightness of the image.

### Outputs

- **transformed:** (torch.tensot) Tensor with the transform applied.

### Source code

```python
def brightness(i, increase = 0.1, prob = 0.5):
    """
        Function to augment images it increases or decreases the brightness of the image by a value of 0 to 20%
    """
    if i.unique().shape[0] != 2: # Hard code to avoid the transform to be done to the GT
        p = np.random.random(1)
        if p < prob:
            p_inc = np.random.random(1)
            i = i*(1 + increase*p_inc)
            i[i>1] = 1.0
        else:
            p_dec = np.random.random(1)
            i = i*((1 - increase*p_dec))

    return i.float()
```

## get_transforms()

Function that returns the transforms to be made on the fly to the cashew dataset.

### Params

### Outputs

### Dependencies used

```python
import torchvision.transforms.v2 as T
```

### Source code

```python
def get_transforms():
    """
        Function that will return the transform to be made on the fly to data.
    """
    transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.Lambda(brightness)
    ])
    return transform
```