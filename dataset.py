#%%
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
from torch_geometric.data import InMemoryDataset, Data, DataLoader


SPACE = 100
BOND_TH = 6.0
#%%      

class PDBBindCoor(InMemoryDataset):

    def __init__(self, root, args, split='train', data_type='coor2', transform=None,
                 pre_transform=None, pre_filter=None):

        
        self.input_list = args.input_list
        self.exp_list = args.exp_list
        self.split = split
        self.data_type = data_type
        self.edge_dim = args.edge_dim
        self.processed = args.processed     
        super().__init__(root, transform, pre_transform, pre_filter)
        path = os.path.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        if self.data_type == 'autodock':
            return ['test']
        return [self.split]
        return [
            'train', 'test'
        ]

    @property
    def processed_dir(self):

        return os.path.join(self.root, self.processed)

    @property
    def processed_file_names(self):
        if self.data_type == 'autodock':
            return ['test.pt']
        return [self.split+'.pt']
        return ['train.pt', 'test.pt']

    # TODO: implement after uploading the entire dataset somewhere
    def download(self):
        print('Hello Download')
        pass

    def process(self):

        splits = [self.split]
        for split in splits:

            dataset_dir = os.path.join(self.raw_dir, f'{split}') 
            files_num = len(
                glob.glob(os.path.join(dataset_dir, '*_data-G.json'))) 

            data_list = []
            graph_idx = 0

            pbar = tqdm(total=files_num)
            pbar.set_description(f'Processing {split} dataset')
            print(f'dataset_dir: {dataset_dir}')

            input_list = f'{self.input_list}{split}'
            with open(input_list) as f:
                data_set = [line.strip() for line in f]

            exp_list = self.exp_list
            with open(exp_list) as f:
                contents = f.readlines()
            dict_pdb_exp = {k: float(v) for k, v in [s.strip().split(',') for s in contents]} #k=pdbid,v=pka

            edge_dim = self.edge_dim
            
            for f in data_set:
                if not os.path.isfile(f'{dataset_dir}/{f}_data-G.json'):
                    print(f"{dataset_dir}/{f}_data-G.json is not exist!")
                    continue
                with open(os.path.join(dataset_dir, f'{f}_data-G.json')) as gf:
                    graphs = gf.readlines()
                num_graphs_per_file = 1 


                pbar.total = num_graphs_per_file * files_num
                pbar.refresh()

                feat_file = open(os.path.join(dataset_dir, f'{f}_data-feats'), 'rb')

                idx = 0   

                features = np.load(feat_file)

                indptr = ast.literal_eval(graphs[3*idx])
                indices = ast.literal_eval(graphs[3*idx+1]) 
                dist = ast.literal_eval(graphs[3*idx+2]) 

                indptr = torch.LongTensor(indptr)
                indices = torch.LongTensor(indices)

                dist = torch.tensor(dist, dtype=torch.float)
                row_idx = torch.ops.torch_sparse.ind2ptr(indptr,len(indices))[1:]

                edge_index = torch.stack((row_idx, indices), dim=0) 
                
                x = torch.Tensor(features)

                labels = dict_pdb_exp[f]

                y = torch.tensor([labels])

                dist = dist.reshape(dist.size()[0], edge_dim) 

                
                data = Data(x=x, edge_index=edge_index, y=y) 
                data.dist = dist

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)
            
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list), os.path.join(self.processed_dir, f'{split}.pt'))

class PDBBindCoor_holo_apo(InMemoryDataset):

    def __init__(self, root, args, split='train', data_type='coor2', transform=None,
                 pre_transform=None, pre_filter=None):
        #self.input_list = input_list
        
        self.input_list = args.input_list
        self.exp_list = args.exp_list
        self.split = split
        self.data_type = data_type
        self.edge_dim = args.edge_dim
        self.processed = args.processed
        self.classify=args.classify    
        super().__init__(root, transform, pre_transform, pre_filter)
        path = os.path.join(self.processed_dir, f'{split}-wat0.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        if self.data_type == 'autodock':
            return ['test']
        return [self.split]
        return [
            'train', 'test'
        ]

    @property
    def processed_dir(self):
        # name = 'subset' if self.subset else 'full'
        # return os.path.join(self.root, name, 'processed')
        return os.path.join(self.root, self.processed)

    @property
    def processed_file_names(self):
        if self.data_type == 'autodock':
            return ['test-wat0.pt']
        return [self.split+'-wat0.pt']
        return ['train.pt', 'test.pt']

    # TODO: implement after uploading the entire dataset somewhere
    def download(self):
        print('Hello Download')
        pass

    def process(self):

        splits = [self.split]
        for split in splits:
            dataset_dir = os.path.join(self.raw_dir, f'{split}') #'MedusaGraph_tmp/pdbbind_rmsd_srand_coor2/raw/test'
            files_num = len(glob.glob(os.path.join(dataset_dir, '*_data-G-wat0.json'))) # get json-files number

            data_list = []

            pbar = tqdm(total=files_num)
            pbar.set_description(f'Processing {split} dataset')
            print(f'dataset_dir: {dataset_dir}')


            input_list = f'{self.input_list}{split}'
            with open(input_list) as f:
                data_set = [line.strip() for line in f]

            exp_list = self.exp_list
            with open(exp_list) as f:
                contents = f.readlines()
            dict_pdb_exp = {k: float(v) for k, v in [s.strip().split(',') for s in contents]} #k=pdbid,v=pka
            

            edge_dim = 10 

            pdblist=[]  
            for f in data_set:
                    

                if not os.path.isfile(f'{dataset_dir}/{f}_data-G.json') or not os.path.isfile(f'{dataset_dir}/{f}_data-G-wat0.json'):
                    print(f"{dataset_dir}/{f}_data-G.json is not exist!")
                    continue
                
                if os.path.getsize(f'{dataset_dir}/{f}_data-G-wat0.json') < 100:
                    continue
                pdblist.append(f)

                with open(os.path.join(dataset_dir, f'{f}_data-G.json')) as gf:
                    graphs = gf.readlines()
                #apo
                with open(os.path.join(dataset_dir, f'{f}_data-G-wat0.json')) as gf:
                    graphs_apo = gf.readlines()
                num_graphs_per_file = 1 

                pbar.total = num_graphs_per_file * files_num
                pbar.refresh()

                feat_file = open(os.path.join(dataset_dir, f'{f}_data-feats'), 'rb')
                feat_file_apo = open(os.path.join(dataset_dir, f'{f}_data-feats-wat0'), 'rb')

                idx = 0

                features = np.load(feat_file)
                

                indptr = ast.literal_eval(graphs[3*idx])
                indices = ast.literal_eval(graphs[3*idx+1]) 
                dist = ast.literal_eval(graphs[3*idx+2]) 
                indptr = torch.LongTensor(indptr)
                indices = torch.LongTensor(indices)
                dist = torch.tensor(dist, dtype=torch.float)
                row_idx = torch.ops.torch_sparse.ind2ptr(indptr,len(indices))[1:]

                edge_index = torch.stack((row_idx, indices), dim=0) 
                
                x = torch.Tensor(features)

            
                labels = dict_pdb_exp[f]

                if self.classify:
                    y = torch.tensor([int(labels)])
                else:
                    y = torch.tensor([labels])

                dist = dist.reshape(dist.size()[0], 6) # 
            
                #apo
                features = np.load(feat_file_apo)
                indptr = ast.literal_eval(graphs_apo[3*idx])
                indices = ast.literal_eval(graphs_apo[3*idx+1]) 
                dist_apo = ast.literal_eval(graphs_apo[3*idx+2]) 
                indptr = torch.LongTensor(indptr)
                indices = torch.LongTensor(indices)
                dist_apo = torch.tensor(dist_apo, dtype=torch.float)
                row_idx = torch.ops.torch_sparse.ind2ptr(indptr,len(indices))[1:]

                edge_index_apo = torch.stack((row_idx, indices), dim=0) #
                
                x_apo = torch.Tensor(features)
                dist_apo = dist_apo.reshape(dist_apo.size()[0], 10) # 
                #pad
                s_apo = x_apo.size()[0]
                s_holo = x.shape[0]
                s_diff = s_holo -s_apo
                pad=(0,0,0,s_diff)
                x_apo = F.pad(x_apo,pad,'constant',0)   
                data = Data(x=x,edge_index=edge_index, y=y) 

                data.dist = dist
                data.x_apo = x_apo
                data.edge_index_apo = edge_index_apo
                data.dist_apo = dist_apo

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)
            
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list), os.path.join(self.processed_dir, f'{split}-wat0.pt')) 



