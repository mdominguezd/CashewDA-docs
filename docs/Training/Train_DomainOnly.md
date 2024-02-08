---
sidebar_position: 2
---

## Brief description of the submodule

In this submodule, all of the functions used for training the domain-only models are described in detail.

## evaluate()

Function used to evaluate the segmentation performance of a specified network. It gets the predictions calculated using the network for a specified data loader and calculates the mean loss and accuracy comparing the predictions with the ground truth labels of the loader.

### Params
- **net:** (torch.nn.Module) Network class used to get the segmentation predictions.
- **validate_loader:**  (torch.nn.DataLoader) Data loader of which the mean loss and accuracy will be calculated.
- **loss_function:** (torch.nn.Module) Loss function used during the training of the network.
- **accu_function:** (torchmetrics.classification) Function to calculate the mean accuracy of the network on validate_loader. *Default* is BinaryF1Score()

### Outputs

- **metric:** (list) List with the mean values of accuracy and loss.

### Dependencies used

```python
import torch
from torchmetrics.classification import BinaryF1Score

from utils import LOVE_resample_fly
```
### Source code

```python

def evaluate(net, validate_loader, loss_function, accu_function = BinaryF1Score(), Love = False, binary_love = False):
    """
        Function to evaluate the performance of a network on a validation data loader.

        Inputs:
            - net: Pytorch network that will be evaluated.
            - validate_loader: Validation (or Test) dataset with which the network will be evaluated.
            - loss_function: Loss function used to evaluate the network.
            - accu_function: Accuracy function used to evaluate the network.

        Output:
            - metric: List with loss and accuracy values calculated for the validation/test dataset.
    """
    
    net.eval()  # Set the model to evaluation mode
    device = next(iter(net.parameters())).device # Get training device ("cuda" or "cpu")

    f1_scores = []
    losses = []

    with torch.no_grad():
        # Iterate over validate loader to get mean accuracy and mean loss
        for i, Data in enumerate(validate_loader):
            
            # The inputs and GT are obtained differently depending of the Dataset (LoveDA or our own DS)
            if Love:
                inputs = LOVE_resample_fly(Data['image'])
                GTs = LOVE_resample_fly(Data['mask'])
                if binary_love:
                    GTs = (GTs == 6).long()
            else:
                inputs = Data[0]
                GTs = Data[1]
        

            inputs = inputs.to(device)
            GTs = GTs.type(torch.long).squeeze().to(device)
            pred = net(inputs)
        
            f1 = accu_function.to(device)
        
            if (pred.max(1)[1].shape != GTs.shape):
                GTs = GTs[None, :, :]

            loss = loss_function(pred, GTs)/GTs.shape[0]
        
            f1_score = f1(pred.max(1)[1], GTs)
            
            f1_scores.append(f1_score.to('cpu').numpy())
            losses.append(loss.to('cpu').numpy())

        metric = [np.mean(f1_scores), np.mean(losses)]   
        
    return metric
```

## training_loop()

Function to train the neural network through backward propagation.

### Params

 - **train_loader:** DataLoader with the training dataset.
- **val_loader:** DataLoader with the validation dataset.
- **learning_rate:** Initial learning rate for training the network.
- **starter_channels:** Starting number of channels in th U-Net
- **momentum:** Momentum used during training.
- **number_epochs:** Number of training epochs.
- **loss_function:** Function to calculate loss.
- **accu_function:** Function to calculate accuracy (Default: BinaryF1Score).
- **Love:** Boolean to decide between training with LoveDA dataset or our own dataset.
- **decay:** Factor in which learning rate decays.
- **bilinear:** Boolean to decide the upscaling method (If True Bilinear if False Transpose convolution. Default: True)
- **n_channels:** Number of initial channels (Defalut 4 [Planet])
- **n_classes:** Number of classes that will be predicted (Default 2 [Binary segmentation])
- **plot:** Boolean to decide if training loop should be plotted or not.
- **seed:** Seed that will be used for generation of random values.

### Outputs

### Dependencies used



## train_3fold_DomainOnly()


## train_LoveDA_DomainOnly()