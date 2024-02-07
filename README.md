# DWMGrad Optimizer

This repository contains an implementation of the DWMGrad optimizer: 
it relies on historical information to dynamically guide momentum and learning rate updates to adapt to different training scenarios, thereby improving performance in various situations.

## Experiments

### Rosenbrock Function: Classical unconstrained optimization problem
result
<img src="example/Rosenbrock /result/path.png" width="500"> 

### CIFAR-10: Image classification
result
<img src="example/CNN/result/img.png" width="500"> 

### MINST: Image classification
result
<img src="example/Multi Layer Net/result/loss_acc.png" width="500"> 

### Core: Text classification
result
<img src="example/RCN/result/loss-acc.png" width="500">


## performance

|          | CIFAR-10 | MINST | Core | Rosenbrock |
|----------|----------|-------|------|------------|
| other-op | 0        | 0     | 0    | 0          |
| our-op   | 1        | 1     | 1    | 1          |

1 represents a better performance in this experiment.

## More
More details are being worked out.



