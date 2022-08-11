import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from kitti_data import KITTI
from prednet import PredNet

from debug import info


num_epochs = 150
batch_size = 16
A_channels = (3, 48, 96, 192)
R_channels = (3, 48, 96, 192)
lr = 0.001 # if epoch < 75 else 0.0001
nt = 10 # num of time steps

if torch.cuda.is_available():
    device = 'cuda'
    print('Using GPU.')
else:
    device = 'cpu'
    print('Using CPU.')

print('on the device', device)

layer_loss_weights = Variable(torch.FloatTensor([[1.], [0.], [0.],
    [0.]]).to(device))
time_loss_weights = 1./(nt - 1) * torch.ones(nt, 1)
time_loss_weights[0] = 0
time_loss_weights = Variable(time_loss_weights.to(device))

DATA_DIR = './kitti_data'

train_file = os.path.join(DATA_DIR, 'X_train.hkl')
train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
val_file = os.path.join(DATA_DIR, 'X_val.hkl')
val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')

kitti_train = KITTI(train_file, train_sources, nt)
kitti_val = KITTI(val_file, val_sources, nt)

train_loader = DataLoader(kitti_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(kitti_val, batch_size=batch_size, shuffle=True)

#model = PredNet(R_channels, A_channels, output_mode='error')
#model = model.to(device)

model = PredNet(R_channels, A_channels, output_mode='prediction')
model.load_state_dict(torch.load('training.pt'))
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print('the model is,',  model)

def lr_scheduler(optimizer, epoch):
    if epoch < num_epochs //2:
        return optimizer
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001
        return optimizer
1/0
# load test ewights 

# test script stuff
for i, inputs in enumerate(val_loader):
    inputs = inputs.permute(0, 1, 4, 2, 3) # batch x time_steps x channel x width x height
    print('inputs shape we need, ', inputs.shape)

    print('inputs we need,1 ', inputs[1])
    print('inputs we need,1 ', inputs[1].shape)
    inputs = Variable(inputs.to(device))
    origin = inputs.data.cpu().byte()[:, nt-1]
    print('origin:')
    print(type(origin))
    print(origin.size())
    print('origin shape we need', origin.shape)

    1/0
    print('predicted:')
    pred = model(inputs)
    pred = pred.data.cpu().byte()
    print(type(pred))
    print(pred.size())
    origin = torchvision.utils.make_grid(origin, nrow=4)
    pred = torchvision.utils.make_grid(pred, nrow=4)
    save_image(origin, 'origin.jpg')
    save_image(pred, 'predicted.jpg')
    break

1/0

for epoch in range(num_epochs):
    optimizer = lr_scheduler(optimizer, epoch)
    for i, inputs in enumerate(train_loader):
        inputs = inputs.permute(0, 1, 4, 2, 3) # batch x time_steps x channel x width x height
        inputs = Variable(inputs.to(device))
        errors = model(inputs) # batch x n_layers x nt
        loc_batch = errors.size(0)
        errors = torch.mm(errors.view(-1, nt), time_loss_weights) # batch*n_layers x 1
        errors = torch.mm(errors.view(loc_batch, -1), layer_loss_weights)
        errors = torch.mean(errors)

        optimizer.zero_grad()

        errors.backward()

        optimizer.step()


#torch.save(model.state_dict(), 'kitti_training.pt')
