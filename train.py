import os.path as osp

# SAGPool
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import time
from utils import EarlyStopping


from models import Set2SetNet, SortPoolNet, gPoolNet, SAGPoolNet

model_str='SAGPool'
# Set2Set/SortPool/gPool/SAGPool/
data_str='Mutagenicity'
# PROTEINS/DD/NCI1/NCI109/Mutagenicity
epochs=300
epoch_start=0
patience=30


path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'datasets', data_str)
dataset = TUDataset(path, name=data_str)
dataset = dataset.shuffle()

save_path='./model_files/' + model_str +'_'+ data_str +'.pth'

# n = len(dataset) // 10
# test_dataset = dataset[:n]
# train_dataset = dataset[n:]
n = (len(dataset) + 9) // 10
test_dataset = dataset[:n]
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]

test_loader = DataLoader(test_dataset, batch_size=256)
train_loader = DataLoader(train_dataset, batch_size=256)
val_loader = DataLoader(val_dataset, batch_size=256)

model=None

if model_str=='Set2Set':
    model=Set2SetNet

elif model_str=='SortPool':
    model = SortPoolNet

elif model_str=='gPool':
    model = gPoolNet

elif model_str=='SAGPool':
    model = SAGPoolNet



device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
model = model(dataset.num_features, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

if osp.exists(save_path):
    checkpoint =torch.load(save_path)
    model.load_state_dict(state_dict=checkpoint['model_state_dict'])
    epoch_start=checkpoint['epoch']
    print('successfully load model!')
else:
    start_epoch = 0
    print('no model, start new training!')


early_stopping = EarlyStopping(patience=patience, path=save_path, verbose=True)

def train():
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


start_time = time.time()
for epoch in range(epoch_start, epoch_start+epochs+1):

    loss = train()
    train_acc = test(train_loader)
    val_acc = test(val_loader)

    print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Val Acc: {:.5f}'.
          format(epoch, loss, train_acc, val_acc))

    # early_stopping
    early_stopping(val_acc, model,epoch)
    if early_stopping.early_stop:
        print("Early stopping")
        print("Max Val Acc ",early_stopping.val_acc_max)
        break

end_time = time.time()
test_acc=test(test_loader)

torch.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
            }, save_path)
print('Test Acc: {:.5f}'.format(test_acc))
print('time cost: %s',end_time - start_time)

