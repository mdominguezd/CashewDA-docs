## Brief description of the submodule

In this submodule the code used for evaluating the performance of the domain adaptation methods is explained.

## cosine_sim()
Function to calculate cosine simmilarity between features extracted in source and target domain.

### Params
- source_F: Features extracted from source domain
- target_F: Features extracted from target domain

### Outputs
- cos: Mean cosine similarity between features extracted on both domains.

### Dependencies
```py
from torch import nn
```

### Source code
```py
def cosine_sim(source_F, target_F):
    """
        Function to calculate cosine simmilarity between features extracted in source and target domain.

        Input
            - source_F: Features extracted from source domain
            - target_F: Features extracted from target domain

        Output:
            - cos: Numeric value of cosine similarity
    """

    COS = nn.CosineSimilarity(dim = 1, eps = 1e-6)
    
    cos = COS(torch.tensor(source_F), torch.tensor(target_F)).mean()

    return cos.numpy()
```

## euc_dist()

Function to calculate de euclidean distance between a vector of features extracted from source and target domain.

### Params
- source_F: Features extracted from source domain
- target_F: Features extracted from target domain

### Ouptuts
- euc_dist: Numeric value of eulidean distance

### Dependencies used
```py
import torch
```

### Source code

```py
def euc_dist(source_F, target_F):
    """
        Function to calculate de euclidean distance between vector of features extracted from source and target domain.

        Input:
            - source_F: Features extracted from source domain
            - target_F: Features extracted from target domain
        
        Output:
            - euc_dist: Numeric value of eulidean distance
    """
    
    source_F = torch.tensor(source_F)
    target_F = torch.tensor(target_F)
    
    euc_dist = torch.cdist(source_F, target_F).mean()

    return euc_dist
```

## get_features_extracted()
Function to get the features extrcted from a specific neural network on both the source and target domain.

### Params
- source_domain: Source domain from which the features will be extracted
- target_domain: Target domain from which the features will be extracted
- DS_args: List of arguments of the dataset of bothe the source and target domain (i.e. [batch_size, transforms, normalization, Vegetation_Indices, Only_DS True if model has DA])
- network: Torch network
- network_filename: Filename of the model to be used.
- Love: Boolean to indicate the use of LoveDA dataset
- cos_int: Boolean to indicate if the calculation of cosine similarity should be done on the go
- euc_int: Boolean to indicate if the calculation of euclidean distance should be done on the go

### Outputs
- Either a metric if cos_int or euc_int are not False
- Or the Features extracted and sample images
### Dependencies used
```py
import torch
from tqdm import tqdm
import numpy as np

from Dataset.ReadyToTrain_DS import get_LOVE_DataLoaders, get_DataLoaders
from utils import get_training_device
```

### Source code
```py
device = get_training_device()

    if network == None:
        network = torch.load(network_filename, map_location = device) 
    
    if Love:
        source_loaders = get_LOVE_DataLoaders(source_domain, *DS_args)
        target_loaders = get_LOVE_DataLoaders(target_domain, *DS_args)

    else:
        source_loaders = get_DataLoaders(source_domain, *DS_args)
        target_loaders = get_DataLoaders(target_domain, *DS_args)

    n_batches = min(len(source_loaders[0]), len(target_loaders[0])) 

    batches = enumerate(zip(source_loaders[0], target_loaders[0]))

    cos = 0

    if cos_int or euc_int:
        d = 'Calculating distance metric'
    else:
        d = 'Getting features extracted'

    num_imgs = 4
    
    for i in tqdm(range(n_batches), desc = d):

        k, (source, target) = next(batches)

        if Love:
            source_input = LOVE_resample_fly(source['image']).to(device)
            target_input = LOVE_resample_fly(target['image']).to(device)  
        else:
            source_input = source[0].to(device)
            target_input = target[0].to(device)
        
        max_batch_size = np.min([source_input.shape[0], target_input.shape[0]])

        s_features = network.FE(source_input)[:max_batch_size].flatten(start_dim = 1).cpu().detach().numpy()
        t_features = network.FE(target_input)[:max_batch_size].flatten(start_dim = 1).cpu().detach().numpy()

        if i == 0:
            s_imgs = source_input[:num_imgs]
            # s_feats = s_features[:num_imgs]
            t_imgs = target_input[:num_imgs]
            # t_feats = t_features[:num_imgs]

        if cos_int:
            cos += cosine_sim(s_features, t_features)
        elif euc_int:
            cos += euc_dist(s_features, t_features)
        else:
            if i == 0:
                source_F = np.array(s_features)
                target_F = np.array(t_features)
            else:
                source_F = np.append(source_F, s_features, axis = 0)
                target_F = np.append(target_F, t_features, axis = 0)
                

    cos /= n_batches
    
    if cos_int or euc_int:
        return cos
    else:
        return source_F, target_F, s_imgs, t_imgs
```

## tSNE_source_n_target()

Function to visualize the features extracted for source and target domains.

### Params
- source_F: Features extracted from source domain
- target_F: Features extracted from target domain

### Dependencies used

```py
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

### Source code
```py
def tSNE_source_n_target(source_F, target_F):
    """
        Function to visualize the features extracted for source and target domains.
    """

    X = np.append(source_F[:, :200000], target_F[:, :200000], axis = 0)
    
    domains = ['Source'] * source_F.shape[0] + ['Target'] * target_F.shape[0]
    
    comps = 2

    tsne = TSNE(comps, random_state = 123, perplexity = 50)

    tsne_X = tsne.fit_transform(X)

    fig, ax = plt.subplots(1,1,figsize = (7,4.5))

    sns.scatterplot(x = tsne_X[:,0], y = tsne_X[:,1], hue = domains, ax = ax, palette = ['darkblue', 'darkred'])

    plt.tight_layout()

    fig.savefig('t_SNE_simple.png', dpi = 200)
    
    for i in range(4):
        ax.scatter(x = tsne_X[i,0], y = tsne_X[i,1], s = 250, c = 'blue', zorder = -1)
        ax.text(x = tsne_X[i,0], y = tsne_X[i,1], s = str(i+1), color = 'white')
        ax.scatter(x = tsne_X[source_F.shape[0] + i,0], y = tsne_X[source_F.shape[0] + i,1], s = 250, c = 'red', zorder = -1)
        ax.text(x = tsne_X[source_F.shape[0] + i,0], y = tsne_X[source_F.shape[0]+ i,1], s = str(i+1), color = 'white')

    plt.tight_layout()

    fig.savefig('t_SNE.png', dpi = 200)

    return tsne_X
```