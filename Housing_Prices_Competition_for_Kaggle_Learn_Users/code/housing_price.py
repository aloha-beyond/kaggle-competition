import math
import numpy as np
import pandas as pd
import os
import csv
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

def same_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def train_valid_split(data_set, valid_ratio, seed):
    valid_data_size = int(len(data_set) * valid_ratio)
    train_data_size = len(data_set) - valid_data_size
    print(valid_data_size)
    print(train_data_size)
    print(data_set.shape)
    data_set = data_set.numpy()
    train_data, valid_data = random_split(data_set, [train_data_size, valid_data_size], generator=torch.Generator().manual_seed(seed))

    # return train_data, valid_data
    return np.array(train_data), np.array(valid_data)

def select_feat(train_data, valid_data, test_data, select_all = True):
    print(train_data.shape, valid_data.shape, test_data.shape)
    y_train = train_data[:, -1]
    y_valid = valid_data[:, -1]
    raw_x_train = train_data[:, :-1]
    raw_x_valid = valid_data[:, :-1]
    raw_x_test = test_data
    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = [0, 1, 2, 3, 4]
    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid

class HousingDataset(Dataset):
    def __init__(self, features, targets=None):
        if targets is None:
            self.targets = targets
        else:
            print(type(targets))
            self.targets = torch.FloatTensor(targets)
        self.features = torch.FloatTensor(features)

    def __getitem__(self, idx):
        if self.targets is None:
            return self.features[idx]
        else:
            return self.features[idx], self.targets[idx]

    def __len__(self):
        return self.features.shape[0]

class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        self.layerS = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )
    def forward(self, x):
        x = self.layerS(x)
        x = x.squeeze(1)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 1122408,
    'select_all': True,
    'valid_ratio': 0.2,
    'n_epochs': 3000,
    'batch_size': 1000,
    'learning_rate': 1e-1,
    'early_stop': 3000,
    'save_path': './models/model.ckpt'
}

def trainer(train_loader, valid_loader, model, config, device):
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)
    writer = SummaryWriter()
    if not os.path.isdir('./models'):
        os.mkdir('./models')
    n_epochs = config['n_epochs']
    best_loss = math.inf
    step = 0
    early_stop_count = 0
    for epoch in range(n_epochs):
        model.train()
        loss_record = []
        train_pbar = tqdm(train_loader, position=0, leave=True)
        for x, y in train_pbar:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            step += 1
            loss_record.append(loss.detach().item())

            train_pbar.set_description(f'Epoch[{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})
        mean_train_loss = sum(loss_record)/len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval()
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)
            loss_record.append(loss.detach().item())

        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(f'Epoch[{epoch+1}/{n_epochs}]:Train loss:{mean_train_loss:.4f}, Valid loss:{mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])
            print("Saving model with loss{:.3f}...".format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1
        if early_stop_count > config['early_stop']:
            print("Early stopping")
            return




def predict(test_loader, model, device):
    model.eval()
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds

def save_pred(preds, file):
    with open(file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "SalePrice"])
        for i, p in enumerate(preds):
            writer.writerow([i+1461, p])





if __name__ == '__main__':
    same_seed(config['seed'])

    train_data = pd.read_csv('../data/train.csv')
    test_data = pd.read_csv('../data/test.csv')
    print(train_data.shape, test_data.shape)

    train_labels = train_data.iloc[:, -1]
    row_name = train_data.columns[-1]
    train_data.drop(row_name, axis=1, inplace=True)
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
    all_features[numeric_features] = all_features[numeric_features].fillna(0)
    all_features = pd.get_dummies(all_features, dummy_na=True, dtype=np.int32)
    n_train = train_data.shape[0]
    all_features2 = all_features[:n_train]
    all_features2[row_name] = train_labels
    train_features = torch.tensor(all_features2.to_numpy(dtype=np.float32))
    test_features = torch.tensor(all_features[n_train:].to_numpy(dtype=np.float32))

    # train_labels = train_data.iloc[:, -1:]
    # row_name = train_data.columns[-1]
    # all_features = train_data.iloc[:, 1:-1]
    # numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    # all_features[numeric_features] = all_features[numeric_features].fillna(0)
    # all_features = pd.get_dummies(all_features, dummy_na=True, dtype=np.int32)
    # all_features[row_name] = train_labels
    # n_train = train_data.shape[0]
    # train_features = torch.tensor(all_features[:n_train].to_numpy(dtype=np.float32))
    #
    # all_features = test_data.iloc[:, 1:]
    # numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    # all_features[numeric_features] = all_features[numeric_features].fillna(0)
    # all_features = pd.get_dummies(all_features, dummy_na=True, dtype=np.int32)
    # n_train = test_data.shape[0]
    # test_features = torch.tensor(all_features[:n_train].to_numpy(dtype=np.float32))

    train_data = train_features
    test_data = test_features
    train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])
    print(
        f"""train_data size : {train_data.shape}, valid_data size : {valid_data.shape}, test_data size : {test_data.shape})""")
    x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])
    print(f"the number of features:{x_train.shape[1]}")
    train_dataset = HousingDataset(x_train, y_train)
    valid_dataset = HousingDataset(x_valid, y_valid)
    test_dataset = HousingDataset(x_test)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
    model = My_Model(input_dim=x_train.shape[1]).to(device)
    trainer(train_loader, valid_loader, model, config, device)

    del model
    model = My_Model(input_dim=x_train.shape[1]).to(device)
    model.load_state_dict(torch.load(config['save_path']))
    preds = predict(test_loader, model, device)
    save_pred(preds, 'pred.csv')








