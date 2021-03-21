# Orthogonal Projection Loss

Contains code for our paper titled _"Orthogonal Projection Loss"_.  

## Usage
Refer to requirements.txt for dependencies. Orthogonal Projection Loss (OPL) can be simply plugged-in with any standard
loss function similar to Softmax Cross-Entropy Loss (CE) as below. You may need to edit the forward function of your 
model to output features (we use the penultimate feature maps) alongside the final logits. You can set the gamma and 
lambda values to default as 0.5 and 1 respectively. 

```python
import torch.nn.functional as F

from loss import OrthogonalProjectionLoss

ce_loss = F.cross_entropy
op_loss = OrthogonalProjectionLoss(gamma=0.5)
op_lambda = 1

for inputs, targets in dataloader:
    features, logits = model(inputs)

    loss_op = op_loss(features, targets)
    loss_ce = ce_loss(logits, targets)

    loss = loss_ce + op_lambda * loss_op
    loss.backward()
```  

