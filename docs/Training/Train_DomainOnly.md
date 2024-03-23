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
- **Love:** Binary to indicate if working with LoveDA dataset.
- **binary_love:** Binary to indicate if working with only one class of LoveDA dataset.

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
- **best_model:** f1-score of the best model trained. (Calculated on validation dataset) 
- **model_saved:** The best model trained.
- **spearman:** Spearman correlation calculated for training progress (High positive value will indicate positive learning)

### Dependencies used
```py
import numpy as np
import torch
from collections import deque

from utils import LOVE_resample_fly, get_training_device
```

### Source code
```py
def training_loop(network, train_loader, val_loader, learning_rate, momentum, number_epochs, loss_function, accu_function = BinaryF1Score(), Love = False, binary_love = False, decay = 0.75, bilinear = True, n_channels = 4, n_classes = 2, plot = True, seed = 8):
    """
        Function to train the Neural Network.

        Input:
            - train_loader: DataLoader with the training dataset.
            - val_loader: DataLoader with the validation dataset.
            - learning_rate: Initial learning rate for training the network.
            - starter_channels: Starting number of channels in th U-Net
            - momentum: Momentum used during training.
            - number_epochs: Number of training epochs.
            - loss_function: Function to calculate loss.
            - accu_function: Function to calculate accuracy (Default: BinaryF1Score).
            - Love: Boolean to decide between training with LoveDA dataset or our own dataset.
            - decay: Factor in which learning rate decays.
            - bilinear: Boolean to decide the upscaling method (If True Bilinear if False Transpose convolution. Default: True)
            - n_channels: Number of initial channels (Defalut 4 [Planet])
            - n_classes: Number of classes that will be predicted (Default 2 [Binary segmentation])
            - plot: Boolean to decide if training loop should be plotted or not.
            - seed: Seed that will be used for generation of random values.

        Output:
            - best_model: f1-score of the best model trained. (Calculated on validation dataset) 
            - model_saved: The best model trained.
            - spearman: Spearman correlation calculated for training progress (High positive value will indicate positive learning)
    """
    
    device = get_training_device()

    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    network = network
    network.to(device)
    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum = momentum, weight_decay=1e-4)
    
    #Training metrics are computed as a running average of the last x samples
    loss_train = deque(maxlen=len(train_loader))
    accuracy_train = deque(maxlen=len(train_loader))

    val_eps = []
    val_f1s = []
    val_loss = []

    train_eps = []
    train_f1s = []
    train_loss = []
    
    for epoch in tqdm(range(number_epochs), desc = 'Training model'):
    
        #Validation phase 0:
        metric_val = evaluate(network, val_loader, loss_function, accu_function, Love, binary_love)

        val_eps.append(epoch)
        val_f1s.append(metric_val[0])
        val_loss.append(metric_val[1])
            
        #Training phase:
        network.train() #indicate to the network that we enter training mode
        
        for i, Data in enumerate(train_loader): # Iterate over the training dataset and do the backward propagation.
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
            
            #Set the gradients of the model to 0.
            optimizer.zero_grad()
            # Get predictions
            pred = network(inputs)

            if (pred.max(1)[1].shape != GTs.shape):
                GTs = GTs[None, :, :]
            
            loss = loss_function(pred, GTs)
            
            accu = accu_function.to(device)
            accu_ = accu(pred.max(1)[1], GTs)

            loss.backward()

            optimizer.step()
            
            loss_train.append(loss.item()/GTs.shape[0])
            accuracy_train.append(accu_.item())

            train_eps.append(epoch+i/len(train_loader))
            train_f1s.append(np.mean(accuracy_train))
            train_loss.append(np.mean(loss_train))

        #Validation phase 1:
        metric_val = evaluate(network, val_loader, loss_function, accu_function, Love, binary_love)
        print(epoch+1, metric_val)

        val_eps.append(epoch + 1)
        val_f1s.append(metric_val[0])
        val_loss.append(metric_val[1])
        
        if epoch == 0:
            best_model = metric_val[0]
            torch.save(network, 'BestModel.pt')
            model_saved = network
        else:
            if best_model < metric_val[0]:
                best_model = metric_val[0]
                torch.save(network, 'BestModel.pt')
                model_saved = network
        
        if (epoch//10 == epoch/10):
            #After 4 epochs, reduce the learning rate by a factor 
            optimizer.param_groups[0]['lr'] *= decay
            
        if plot:
            fig, ax = plt.subplots(1,1, figsize = (7,5))
    
            ax.plot(train_eps, train_f1s, label = 'Training F1-Score', ls= '--', color = 'r')
            ax.plot(train_eps, train_loss, label = 'Training Loss', ls = '-', color = 'r')
    
            ax.plot(val_eps, val_f1s, label = 'Validation F1-Score', ls = '--', color = 'b')
            ax.plot(val_eps, val_loss, label = 'Validation Loss', ls = '-', color = 'b')
            
            ax.text(val_eps[np.argmax(val_f1s)], np.max(val_f1s), str(np.max(val_f1s)))
    
            ax.set_xlabel("Epoch")
    
            plt.legend()
    
            fig.savefig('TrainingLoop.png', dpi = 200)

            plt.close()

    spearman = stats.spearmanr(val_eps, val_f1s)[0]

    if val_eps[np.argmax(val_f1s)] == 0:
        no_learning = True
    else:
        no_learning = False
        
    return best_model, model_saved, spearman, no_learning
```

## train_3fold_DomainOnly()

### Params
- domain: String with the prefix of the domain to use for training. (Can be either Tanzania or IvoryCoast)
- DS_args: List with all the arguments related to the dataset itself (e.g. batch_size, transforms, normalization and use of vegetation indices)
- network_args: List with arguments used for the network creation (n_classes, bilinear, starter channels, up_layer)
- training_loop_args: List with all the arguments needed to run the training loop (for more information check training_loop funtion.)
- eval_args: List with arguments to evaluate the trained network on the test dataset.
### Outputs
- **Stats:** List with the mean f1 score and its standard deviation

### Dependencies used
```py
import time
import torch
import numpy as np

from Dataset.ReadyToTrain_DS import get_DataLoaders
```

### Source code
```py
def train_3fold_DomainOnly(domain, DS_args, network_args, training_loop_args, eval_args):
    """
        Function to run all Domain Only training for the three folds. 

        Input:
            - domain: String with the prefix of the domain to use for training. (Can be either Tanzania or IvoryCoast)
            - DS_args: List with all the arguments related to the dataset itself (e.g. batch_size, transforms, normalization and use of vegetation indices)
            - network_args: List with arguments used for the network creation (n_classes, bilinear, starter channels, up_layer)
            - training_loop_args: List with all the arguments needed to run the training loop (for more information check training_loop funtion.)
            - eval_args: List with arguments to evaluate the trained network on the test dataset.

        Output:
            - Stats: Mean and standard deviation of the validation and test accuracy values for the domain only training on the three folds.
    """

    folds = 3

    fscore = []
    
    # For 3-fold Cross-Validation
    for i in range(folds):
        
        # Build Dataloaders
        print("Creating dataloaders...")
        train_loader, val_loader, test_loader = get_DataLoaders(domain+'Split'+str(i+1), *DS_args)
        print("Dataloaders created.\n")
        
        n_channels = next(enumerate(train_loader))[1][0].shape[1] #get band number from actual data
        n_classes = 2
        
        # Define the network
        network = UNet(n_channels, *network_args)
        
        # Train the model
        print("Starting training...")
        start = time.time()
        f1_val, network_trained, spearman, no_L = training_loop(network, train_loader, val_loader, *training_loop_args)
        print("Network trained. Took ", round(time.time() - start, 0), 's\n')

        if i == 0:
            best_network = network_trained
            torch.save(best_network, 'OverallBestModel'+domain+'.pt')
            best_f1 = f1_val
        else:
            if f1_val > best_f1:
                best_network = network_trained
                torch.save(best_network, 'OverallBestModel'+domain+'.pt')
                best_f1 = f1_val
        
        # Evaluate the model
        f1_test, loss_test = evaluate(network_trained, test_loader, *eval_args)
        
        print("F1_Validation:", f1_val)
        print("F1_Test:      ", f1_test)
    
        fscore.append([f1_val, f1_test])
    
    fscore
    
    mean = np.mean(fscore, axis = 0)
    std = np.std(fscore, axis = 0)

    stats = [mean, std]

    return stats
```

## train_LoveDA_DomainOnly()

Function to train the domain only models for the LoveDA dataset.

### Params
- domain: List with the scene parameter for the LoveDa dataset. It can include 'rural' and/or 'urban'.
- DS_args: List with all the arguments related to the dataset itself (e.g. batch_size, transforms)
- network_args: List with arguments used for the network creation (n_classes, bilinear, starter channels, up_layer)
- training_loop_args: List with all the arguments needed to run the training loop (for more information check training_loop funtion.)

### Outputs
- validation_accuracy: Accuracy score for validation dataset.
- network_trained: Neural network that has been trained

### Dependencies used
```py
from Dataset.ReadyToTrain_DS import get_LOVE_DataLoaders
```

### Source code
```py
def train_LoveDA_DomainOnly(domain, DS_args, network_args, training_loop_args):
    """
        Function to train the domain only models for the LoveDA dataset.

        Inputs:
            - domain: List with the scene parameter for the LoveDa dataset. It can include 'rural' and/or 'urban'.
            - DS_args: List with all the arguments related to the dataset itself (e.g. batch_size, transforms)
            - network_args: List with arguments used for the network creation (n_classes, bilinear, starter channels, up_layer)
            - training_loop_args: List with all the arguments needed to run the training loop (for more information check training_loop funtion.)

        Outputs:
            - validation_accuracy: Accuracy score for validation dataset.
            - network_trained: Neural network that has been trained.
    """

    # Get DataLoaders
    train_loader, val_loader, test_loader = get_LOVE_DataLoaders(domain, *DS_args)

    # Get number of channels from actual data
    n_channels = next(enumerate(train_loader))[1]['image'].shape[1] 

    # Define the network
    network = UNet(n_channels, *network_args)

    # Train the network
    accu_val, network_trained, spearman, no_l = training_loop(network, train_loader, val_loader, *training_loop_args)

    return accu_val, network_trained
```

## run_DomainOnly()

Aggregating function to perform the whole training routine for one of the domains.

### Params
- **domain:** String with the name of the domain of interest ('Tanzania' or 'IvoryCoast')
### Outputs

### Source code
```py
def run_DomainOnly(domain = 'IvoryCoast'):
    """
        Function to perform the whole training routine for one of the domains.
    """
    
    ## Related to DS
    batch_size = 4
    transforms = get_transforms()
    normalization = 'Linear_1_99'
    VI = True
    DA = False
    
    ## Related to the network
    n_classes = 2
    bilinear = True
    starter_channels = 16
    up_layer = 4
    attention = True
    resunet = False
    
    ## Related to training and evaluation
    number_epochs = 30
    learning_rate = 1
    momentum = 0.2
    loss_function = FocalLoss(gamma = 2)
    accu_function = BinaryF1Score()
    device = get_training_device()
    
    DS_args = [batch_size, transforms, normalization, VI, DA, None, None]
    network_args = [n_classes, bilinear, starter_channels, up_layer, attention, resunet]
    training_args = [learning_rate, momentum, number_epochs, loss_function]
    eval_args = [loss_function, accu_function]
    
    Stats = train_3fold_DomainOnly(domain, DS_args, network_args, training_args, eval_args)

    print(Stats)
    
    plot_3fold_accuracies(domain, Stats)
```