import os
import ast
import argparse
import glob
import torch
import torch.nn.functional as F
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import distance

from tqdm import tqdm
from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
import time

from model import Net_holo, Net_holo_apo
from dataset import PDBBindCoor,PDBBindCoor_holo_apo

from time import time 
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from scipy import stats
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--edge_dim", help="dimension of edge feature", type=int, default = 6)
parser.add_argument("--n_graph_layer", help="number of GNN layer", type=int, default = 4)
parser.add_argument("--d_graph_layer", help="dimension of GNN layer", type=int, default = 256)
parser.add_argument("--dropout_rate", help="dropout rate", type=float, default=0.3)
parser.add_argument("--data_path", help="raw path", type=str, default='Graph_tmp/pdbbind_coor2')
parser.add_argument("--input_list", help="list of train/test pdbs", type=str, default='data/pdb_list/pdb_list_')
parser.add_argument("--exp_list", help="list of pka", type=str, default='data/pdb_list/exp_list.txt')
parser.add_argument("--batch_size", help="batch_size", type=int, default = 64)
parser.add_argument("--output", help="training NN data", type=str, default='./out.txt')
parser.add_argument("--pred_output", help="prediction data", type=str, default='./prediction.txt')
parser.add_argument("--epoch", help="epoch", type=int, default = 2000)
parser.add_argument("--processed", help="processed dir", type=str, default = 'processed')
parser.add_argument("--graph_pooling", help="pooling methods: ", type=str, default = 'sum')
parser.add_argument("--apo", help="water network in apo state", default=False, action='store_true')
parser.add_argument("--gpu_id", help="id of gpu", type=int, default = 0)
parser.add_argument("--pre_model", help="path to trained model checkpoint", type=str, default='None')
parser.add_argument("--test_mode", help="whether we only test the model", default=False, action='store_true')
parser.add_argument("--classify", help="whether we select classification task", default=False, action='store_true')
#args = parser.parse_args(['--d_graph_layer', '256', '--n_graph_layer', '4', '--edge_dim', '6','--dropout_rate', '0.3','--data_path','apo_tmp/pdbbind_coor2','--output','./output-apo.txt','--pred_output','./prediction-apo.txt','--batch_size','64','--processed','processed', '--graph_pooling','sum','--pre_model','apo_tmp/pdbbind_coor2/processed/pred_2_069.pt', '--test_mode', '--apo']) # 
args = parser.parse_args()

SPACE = 100
BOND_TH = 6.0

path = args.data_path
isapo = args.apo
if isapo:
    if not args.test_mode:
        train_dataset=PDBBindCoor_holo_apo(path, args, split='train')
    test_dataset=PDBBindCoor_holo_apo(path, args, split='test') 
else:
    if not args.test_mode:
        train_dataset=PDBBindCoor(path, args, split='train')
    test_dataset=PDBBindCoor(path, args, split='test')    

if not args.test_mode:
    train_loader=DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    train_loader_size = len(train_loader.dataset)
test_loader=DataLoader(test_dataset, batch_size=1)


test_dataset_size = len(test_dataset)
test_loader_size = len(test_loader.dataset)

gpu_id = str(args.gpu_id)
device_str = 'cuda:' + gpu_id if torch.cuda.is_available() else 'cpu'

device = torch.device(device_str)
print('cuda' if torch.cuda.is_available() else 'cpu')

if isapo:
    print('Considering apo')
    model = Net_holo_apo(test_dataset.num_features, args).to(device)

else:
    model = Net_holo(test_dataset.num_features, args).to(device)
if args.pre_model != 'None':
    if torch.cuda.is_available():

        checkpoint= torch.load(args.pre_model)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)

    else:
        checkpoint = torch.load(args.pre_model, map_location=device)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])    

loss_op = torch.nn.MSELoss(reduction='mean').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,eps=1e-3)

def train():
    model.train()

    total_loss = 0
    tot = 0
    t = time()
    pbar = tqdm(total=train_loader_size)
    pbar.set_description('Training poses...')
    for data in train_loader:
        with torch.cuda.amp.autocast():
            data = data.to(device)
            
            optimizer.zero_grad()
            if isapo:
                print('Considering apo in training')
                pred = model(data.x, data.edge_index, data.dist, data.batch, data.x_apo, data.edge_index_apo, data.dist_apo)
            else:
                pred = model(data.x, data.edge_index, data.dist, data.batch)

            loss = loss_op(pred, data.y.unsqueeze_(1))
            assert torch.isnan(loss).sum() == 0, print('LOSS:',loss) 

            total_loss += loss.item() * 1

        loss.backward()
        optimizer.step()
        tot += 1
        pbar.update(1)

    pbar.close()
    
    print(f"trained {tot} batches, take {time() - t}s")
    return total_loss / train_loader_size

@torch.no_grad()
def test(loader, epoch):
    model.eval()
    t = time()

    total_loss = 0
    pred_out = []
    label_out = []


    pose_idx = 0

    pbar = tqdm(total=test_loader_size)
    pbar.set_description('Testing poses...')
    for data in loader:
        data = data.to(device)
        pbar.update(1)
  
        if isapo:
            print('Considering apo')
            pred = model(data.x, data.edge_index, data.dist, data.batch, data.x_apo,data.edge_index_apo,data.dist_apo)
        else:
            pred = model(data.x, data.edge_index, data.dist, data.batch)

        loss = loss_op(pred, data.y.unsqueeze_(1))

        total_loss += loss.item()
        pred_out.append(round(float(pred),2))
        label_out.append(round(float(data.y),2))
        pose_idx += 1

    
    pbar.close()
    tt = time() - t
    print(f"Spend {tt}s")

    return total_loss / pose_idx, pred_out, label_out



def evaluation_method(y_test, y_test_pred):
    try:
        test_pears = pearsonr(y_test, y_test_pred)[0]
        test_spearman=stats.spearmanr(y_test, y_test_pred)[0]
        test_rmse = mean_squared_error(y_test, y_test_pred)**0.5
        return f'{test_pears:.4f}', f'{test_spearman:.4f}', f'{test_rmse:.4f}'
    except:
        print("Error! Inconsistent data")
        return 'NaN', 'NaN', 'NaN'
    
if not args.test_mode:
    min_loss = 9999.99
    for epoch in range(0, args.epoch):
        loss_train = train()

        loss_test,pred,label = test(test_loader, epoch)
        pears,spearm,rmse = evaluation_method(label, pred)
        output = args.output
        if output != 'none':
            with open(output, 'a') as f:
                f.write(f'Epoch: {epoch+1:03d}, Train loss: {loss_train:.4f}, Test loss: {loss_test:.4f}\n')
        pred_output = args.pred_output
        if pred_output != 'none':
            with open(pred_output, 'a') as f: 
                f.write(f'Epoch: {epoch+1:03d}, pearson: {pears}, rmse: {rmse}, spearman: {spearm}\n{label}\n{pred}\n')    
        if epoch > 3 and loss_test < min_loss:
            state = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch':epoch}
            torch.save(state, f'{path}/{args.processed}/{os.path.splitext(os.path.basename(pred_output))[0]}_{epoch+1:03d}.pt')
            min_loss = loss_test
else:
    assert args.pre_model != 'None'
    epoch = 1
    loss_test,pred,label = test(test_loader, epoch)
    with open(args.pred_output, 'a') as f:
        f.write(f'Score:  {pred}\n') 
