from densenet_inter3 import DenseNet as DenseNetInter
from densenet_inter3 import _DenseBlock, Intermediate
from utils import train, count_parameters, mkdir
from torchvision import datasets, transforms
import torch
import torch.nn.utils.prune as prune
import os, time

data = './data'
save = './models/prune/100_bc_bychannel/'

val_size = 5000

mean = [0.5071, 0.4867, 0.4408]
std = [0.2675, 0.2565, 0.2761]

train_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=mean, std=std)])

test_transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=mean, std=std)])

train_set = datasets.CIFAR10(data, train=True, transform=train_transforms, download=True)
test_set = datasets.CIFAR10(data, train=False, transform=test_transforms)
valid_set = datasets.CIFAR10(data, train=True, transform=test_transforms)
indices = torch.randperm(len(train_set))
train_indices = indices[:len(indices) - val_size]
valid_indices = indices[len(indices) - val_size:]
train_set = torch.utils.data.Subset(train_set, train_indices)
valid_set = torch.utils.data.Subset(valid_set, valid_indices)


# ### Model Definition


growth_rate = 12                    # a single number or list of numbers eg: 12 or [12, 24, 48]
bottleneck = True                   # set True or False accordingly
num_init_features = 2*growth_rate   # according to densenet paper, change otherwise
num_classes = 10                    # number of classes to predict
compression = 0.5                   # compression parametere, in case of BC set value to 0.5 for no BC set value to 1

depth = 100                          #depth of network if you want block_config to be decided automatically
N = (depth - 4) // 3
if bottleneck:
    N //= 2
block_config = [N] * 3              # change this to a custom list if you want eg: [9, 16, 25]

def connections(l, num_layers):
    n = int(num_layers**0.5)
    if l >= n and l-n != l-1:
        return [l-1, l-n]
    return [l-1]

connection_config = None            # set it to a function if you want to use same connectivity for all block
                                    # or set it to a list of functions [connection1, connection2, connestion3]
                                    # None means use default densenet connectivity

intermediate_conv_flag = True      # set True to add weights between connections


# do not edit any of the remaining code

model = DenseNetInter(growth_rate=growth_rate,
                     block_config=block_config,
                     num_init_features=num_init_features,
                     num_classes=num_classes,
                     compression=compression,
                     bottleneck=bottleneck,
                     memory_efficient=False,
                     small_inputs=True,
                     connection_config=connection_config,
                     intermediate_conv_flag=True)
print(model)

print('Total trainable parameters:', count_parameters(model))

prune_params = []
for name, module in model.named_modules():
    if isinstance(module, Intermediate):
        prune_params.append((module, 'weight'))

n_epochs = 100
batch_size = 64
prune_factor = 0.05

tms = []

for i in range(1,8):
    save_c = save + str(i)
    print(save_c)
    mkdir(save_c)
    start = time.time()
    train(model=model, train_set=train_set, valid_set=valid_set, test_set=test_set, save=save_c,
        n_epochs=n_epochs, batch_size=batch_size)
    end = time.time()
    print('Training completed in', str(end-start), 'seconds')
    tms.append(end-start)
    if i == 7:
        break
    print('Pruning', i*prune_factor*100,"% intermediate weights")
    prune.global_unstructured(prune_params,
                              pruning_method=prune.L1Unstructured,
                              amount=prune_factor*i
                             )
print(tms)