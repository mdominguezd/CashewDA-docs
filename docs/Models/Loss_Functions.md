## Brief description of the submodule

In this submodule all of the classes used for the definition of the loss functions used during training of the networks described in [Networks implemented](./U_Net#networks-implemented).
## FocalLoss

Class used to calculate the [focal loss](https://paperswithcode.com/method/focal-loss) used for the backward propagation of the segmentation heads of the [Networks implemented](./U_Net#networks-implemented).

The Focal loss can be calculated as:

$$
Focal\ loss = -\alpha \cdot (1-p)^\gamma \cdot \log(p)
$$

Big shout out to [f1recracker](https://gist.github.com/f1recracker/0f564fd48f15a58f4b92b3eb3879149b), whose implementation was adapted to create the focal loss function.

### Attributes

- **gamma:** (float) Hyper-parameter to be tuned which determines how much importance is given to misclassified examples. The higher the value, the higher the importance.
- **alpha:** (float) Hyper-parameter to be tuned which serves as a weighing factorfor the different classes.
- **size_average:** (boolean) Boolean to indicate if the funcion returns the mean value or the sum. Default is True.
- **ignore_index:** (int) Index of the class that will be ignored.

### Methods

- **forward:** Function indicating the forward stepss to calculate the Focal Loss function for an predicted input and a target tensor.


### Source code

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True, ignore_index = None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, input, target):

        if input.dim() > 2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        
        target = target.view(-1,1)
        
        if self.ignore_index != None:
            # Filter predictions with ignore label from loss computation
            mask = target != self.ignore_index

            target = target[mask[:,0], :]
            input = input[mask[:,0], :]
        
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
```

