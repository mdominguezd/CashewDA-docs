## Brief description of the submodule

In this submodule the functions used for the selection of hyper-parameters for the training of the [Networks implemented](./U_Net#networks-implemented).

## HP_Tuning()

Function to execute the hyper-parameter tuning for the [UNet](../Models/U_Net#unet) model on the Cashew Dataset.

It receives the possible values of the hyperparameters as lists and returns a dataframe with the results of each possible combination. 

The metrics considered are:
- **Validation mIOU:** Highest value of mIOU obtained during training.
- **Training time:** Time spent on training.
- **Training rho:** Spearman coefficient to check that the training accuracy is continuously increasing with the epochs.
- **NO Learning:** Boolean indicating if accuracy did improve compared to the one calculated in epoch 0.

The calculation of each of the metrics is done using **12 epochs** and a **Linear normalization** of the Cashew dataset. For more information of this dataset go [here](../Dataset/ReadyToTrain_DS).

### Params
- **dir:** (dir) Directory with the dataset to be used.
- **BS:** (list) List with values of batch_size to be considered during HP tuning.
- **LR**: (list) List with values of learning rate to be considered during HP tuning.
- **STCh:** (list) List with values of starting number of channels to be considered during HP tuning.
- **mu:** (list) List with values of momentum to be considered during HP tuning.
- **Bi:** (llist) List with values of bilinear to be considered during HP tuning. (Only True or False possible)
- **gamma:** (list) List with values of gamma vlaues for the focal loss to be considered during HP tuning.
- **VI:** (list) List with values of vegetation indices (True or False)
- **decay:** (list) List with values of the decay rate of learning rate.
- **atts:** (list) List with booleans for inclusion or not of Attention gates.
- **res:** (list) List with booleans for inclusion or not of residual connections on the double convolutional blocks.
### Outputs

- **HP_values:** (pandas.DataFrame) Dataframe with the results of each iteration of the hyperparameter tuning.

### Dependencies used

```python
import pandas as pd
import time
from torchmetrics.classification import BinaryF1Score

from Dataset.Transforms import getTransforms
from Dataset.ReadyToTrain_DS import getDataLoaders
from Models.U_Net import UNet
from Models.Loss_Functions import FocalLoss
```

### Source code

```python
def HP_Tuning(dir, BS, LR, STCh, MU, Bi, gamma, VI, decay, atts, res, tr_size = 0.15, val_size = 0.75):
    """
        Function to perform Hyperparameter tuning for the networks to be trained.

        Input:
            - dir: Directory with the dataset to be used.
            - BS: List with values of batch_size to be considered during HP tuning.
            - LR: List with values of learning rate to be considered during HP tuning.
            - STCh: List with values of starting number of channels to be considered during HP tuning.
            - mu: List with values of momentum to be considered during HP tuning.
            - Bi: List with values of bilinear to be considered during HP tuning. (Only True or False possible)
            - gamma: List with values of gamma vlaues for the focal loss to be considered during HP tuning.
            - VI: List with values of vegetation indices (True or False)
            - decay: decay rate of learning rate.
            - atts: Inclusion or not of Attention gates.
            - res: Inclusion or not of residual connections on convolutional blocks.
            
        Output:
            - HP_values: (pandas.DataFrame) Dataframe with the results of each iteration of the hyperparameter tuning.
    """

    transforms = get_transforms()
    normalization = 'Linear_1_99'
    epochs = 12

    rows = []

    for bs in BS:
        for lr in LR:
            for stch in STCh:
                for mu in MU:
                    for bi in Bi:
                        for g in gamma:
                            for vi in VI:
                                for de in decay:
                                    for at in atts:
                                        for re in res:
                                            train_loader, val_loader, test_loader = get_DataLoaders(dir, bs, transforms, normalization, vi, train_split_size = tr_size, val_split_size = val_size)
                                            n_channels = next(enumerate(train_loader))[1][0].shape[1] #get band number fomr actual data
                                            n_classes = 2
                
                                            loss_function = FocalLoss(gamma = g)
                
                                            # Define the network
                                            network = UNet(n_channels, n_classes,  bi, stch, up_layer = 4, attention = at, resunet = re)
            
                                            start = time.time()
                                            f1_val, network_trained, spearman, no_l = training_loop(network, train_loader, val_loader, lr, mu, epochs, loss_function, decay = de, plot = False)
                                            end = time.time()
            
                                            rows.append([bs, lr, stch, mu, bi, g, vi, de, at, re, f1_val, end-start, spearman, no_l])
            
                                            HP_values = pd.DataFrame(rows)
                                            HP_values.columns = ['BatchSize','LR', 'StartCh', 'Momentum', 'Bilinear', 'gamma', 'VI', 'decay', 'attention', 'resnet', 'ValF1Score', 'Training time', 'Training rho', 'No_L']
                                            HP_values.to_csv('TempHyperParamTuning_'+dir+'.csv')
    
    return HP_values
```
## LoveDA_HP_Tuning()

Function to execute the hyper-parameter tuning for the [UNet](../Models/U_Net#unet) model on the **LoveDA** dataset

It receives the possible values of the hyperparameters as lists and returns a dataframe with the results of each possible combination. 

The metrics considered are:
- **Validation mIOU:** Highest value of mIOU obtained during training.
- **Training time:** Time spent on training.
- **Training rho:** Spearman coefficient to check that the training accuracy is continuously increasing with the epochs.
- **NO Learning:** Boolean indicating if accuracy did improve compared to the one calculated in epoch 0.

The calculation of each of the metrics is done using **15 epochs**.

### Params

- **BS:** (list) List with values of batch_size to be considered during HP tuning.
- **LR**: (list) List with values of learning rate to be considered during HP tuning.
- **STCh:** (list) List with values of starting number of channels to be considered during HP tuning.
- **mu:** (list) List with values of momentum to be considered during HP tuning.
- **Bi:** (llist) List with values of bilinear to be considered during HP tuning. (Only True or False possible)
- **gamma:** (list) List with values of gamma values for the focal loss to be considered during HP tuning.
- **decay:** (list) List with values of the decay rate of learning rate.
- **atts:** (list) List with booleans for inclusion or not of Attention gates.
- **res:** (list) List with booleans for inclusion or not of residual connections on the double convolutional blocks

### Outputs

- **HP_values:** (pandas.DataFrame) Dataframe with the results of each iteration of the hyperparameter tuning.

### Dependencies used

```python
import pandas as pd
import time
from torchmetrics.classification import BinaryF1Score

from Dataset.Transforms import getTransforms
from Dataset.ReadyToTrain_DS import get_LOVE_DataLoaders
from Models.U_Net import UNet
from Models.Loss_Functions import FocalLoss
```

### Source code
```python
def LoveDA_HP_Tuning(domain, BS, LR, STCh, MU, Bi, gamma, decay, atts, res, tr_size = 0.15, val_size = 0.75):
    """
        Function to perform Hyperparameter tuning for the networks to be trained on LoveDA dataset.

        Input:
            - dir: Directory with the dataset to be used.
            - BS: List with values of batch_size to be considered during HP tuning.
            - LR: List with values of learning rate to be considered during HP tuning.
            - STCh: List with values of starting number of channels to be considered during HP tuning.
            - mu: List with values of momentum to be considered during HP tuning.
            - Bi: List with values of bilinear to be considered during HP tuning. (Only True or False possible)
            - gamma: List with values of gamma vlaues for the focal loss to be considered during HP tuning.
            - decay: decay of learning rate.
            - atts: Boolean indicating if attention gates are used or not.
            - res: Boolean indicating if residua connections on convolutional blocks are used or not.
            
        Output:
            - HP_values: (pandas.DataFrame) Dataframe with the results of each iteration of the hyperparameter tuning.
    """

    transforms = get_transforms()
    # normalization = 'Linear_1_99'
    epochs = 15

    rows = []

    for bs in BS:
        for lr in LR:
            for stch in STCh:
                for mu in MU:
                    for bi in Bi:
                        for g in gamma:
                            for de in decay:
                                for at in atts:
                                    for re in res:
                                        train_loader, val_loader, test_loader = get_LOVE_DataLoaders(domain, bs, train_split_size = tr_size, val_split_size = val_size)
                                        n_channels = next(enumerate(train_loader))[1]['image'].shape[1] #get band number fomr actual data
                                        n_classes = 8
            
                                        loss_function = FocalLoss(gamma = g, ignore_index = 0)
            
                                        # Define the network
                                        network = UNet(n_channels, n_classes,  bi, stch, up_layer = 4, attention = at, resunet = re)
        
                                        start = time.time()
                                        f1_val, network_trained, spearman, no_l = training_loop(network, train_loader, val_loader, lr, mu, epochs, loss_function, decay = de, plot = False, accu_function=JaccardIndex(task = 'multiclass', num_classes = n_classes, ignore_index = 0) , Love = True)
                                        end = time.time()
        
                                        rows.append([bs, lr, stch, mu, bi, g, de, at, re, f1_val, end-start, spearman, no_l])
        
                                        HP_values = pd.DataFrame(rows)
                                        HP_values.columns = ['BatchSize','LR', 'StartCh', 'Momentum', 'Bilinear', 'gamma', 'decay', 'attention', 'resunet', 'ValF1Score', 'Training time', 'Training rho', 'No_L']
                                        HP_values.to_csv('TempHyperParamTuning_LOVE.csv')
    
    return HP_values
```
