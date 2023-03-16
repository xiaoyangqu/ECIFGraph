#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TransformerConv, global_mean_pool, global_add_pool, global_max_pool,Set2Set
import numpy as np
#%%

class Net_screen(torch.nn.Module): #v14
    def __init__(self, num_features, args):
        super().__init__()
        print("get the model of Net_holo_apo")

        d_layer = int(args.d_graph_layer)
        self.conv1 = TransformerConv(num_features, args.d_graph_layer, edge_dim = args.edge_dim)
        self.conv2 = TransformerConv(20, d_layer, edge_dim = 10)

        self.convs = torch.nn.ModuleList([TransformerConv(args.d_graph_layer, args.d_graph_layer, edge_dim = args.edge_dim) for _ in range(args.n_graph_layer)])
        self.convs2 = torch.nn.ModuleList([TransformerConv(d_layer, d_layer, edge_dim = 10) for _ in range(args.n_graph_layer)])
        # self.convs = torch.nn.ModuleList([torch.nn.Linear(args.d_graph_layer, args.d_graph_layer) for _ in range(args.n_graph_layer)])
        self.bn = torch.nn.BatchNorm1d(args.d_graph_layer)
        self.bn2 = torch.nn.BatchNorm1d(d_layer)
        self.convl = TransformerConv(args.d_graph_layer, args.d_graph_layer, edge_dim = args.edge_dim)
        self.convl2 = TransformerConv(d_layer, d_layer, edge_dim = 10)
        # self.convl = torch.nn.Linear(args.d_graph_layer, 3)

        self.fc = nn.Linear(args.d_graph_layer+d_layer,1)
        self.relu = torch.nn.ReLU()
        self.elu = torch.nn.ELU()
        self.gelu = torch.nn.GELU()
        self.leakyrelu = torch.nn.LeakyReLU()

        self.Dropout = torch.nn.Dropout(p=args.dropout_rate)

        graph_pooling = args.graph_pooling
        
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "set2set":
            self.pool = Set2Set(args.d_graph_layer, processing_steps=2)
        if graph_pooling == "set2set":
            self.linl = nn.Linear((args.d_graph_layer+d_layer)*4, args.d_graph_layer+d_layer)
        else:
            self.linl = nn.Linear(args.d_graph_layer+d_layer, int((args.d_graph_layer+d_layer)))
            self.linl2 = nn.Linear(int((args.d_graph_layer+d_layer)), int(0.5*(args.d_graph_layer+d_layer)))
            self.fc = nn.Linear(int(0.5*(args.d_graph_layer+d_layer)),2)


        self.pool2 = global_mean_pool
        #self.pool2 = global_max_pool
        #self.pool2 = global_add_pool


    def forward(self, x, edge_index, edge_attr, batchs, x_apo, edge_index_apo, edge_attr_apo):
        x = self.conv1(x, edge_index, edge_attr)

        x = self.relu(x)
        x = self.Dropout(x)

        for i in range(len(self.convs)):
            #print('Layer:',i)

            x = self.convs[i](x, edge_index, edge_attr)
            # x = self.convs[i](x)
            x = self.relu(x)
            #x = self.bn(x)
            x = self.Dropout(x)


        x = self.pool(x,batchs)

        #apo#
        x_apo = self.conv2(x_apo, edge_index_apo, edge_attr_apo)

        x_apo = self.relu(x_apo)
        x_apo = self.Dropout(x_apo)

        for i in range(len(self.convs2)):
            #print('Layer:',i)

            x_apo = self.convs2[i](x_apo, edge_index_apo, edge_attr_apo)

            x_apo = self.relu(x_apo)
            #x = self.bn2(x_apo)

            x_apo = self.Dropout(x_apo)

        #x_apo = self.convl2(x_apo, edge_index_apo, edge_attr_apo)

        x_apo = self.pool2(x_apo, batchs)
        #x_apo = torch.mean(x_apo,dim=0,keepdim=True)

        #concat
        xc = torch.cat((x, x_apo), 1)
        #readout
        xc = self.linl(xc)
        xc = self.relu(xc)
        xc = self.Dropout(xc)
        xc = self.linl2(xc)
        xc = self.relu(xc)
        xc = self.Dropout(xc)         
        xc = self.fc(xc)
        return F.softmax(xc, dim=1)

class Net_holo_apo(torch.nn.Module): #v14
    def __init__(self, num_features, args):
        super().__init__()
        print("get the model of Net_holo_apo")

        d_layer = int(args.d_graph_layer)
        self.conv1 = TransformerConv(num_features, args.d_graph_layer, edge_dim = args.edge_dim)
        self.conv2 = TransformerConv(20, d_layer, edge_dim = 10)

        self.convs = torch.nn.ModuleList([TransformerConv(args.d_graph_layer, args.d_graph_layer, edge_dim = args.edge_dim) for _ in range(args.n_graph_layer)])
        self.convs2 = torch.nn.ModuleList([TransformerConv(d_layer, d_layer, edge_dim = 10) for _ in range(args.n_graph_layer)])
        # self.convs = torch.nn.ModuleList([torch.nn.Linear(args.d_graph_layer, args.d_graph_layer) for _ in range(args.n_graph_layer)])
        self.bn = torch.nn.BatchNorm1d(args.d_graph_layer)
        self.bn2 = torch.nn.BatchNorm1d(d_layer)
        self.convl = TransformerConv(args.d_graph_layer, args.d_graph_layer, edge_dim = args.edge_dim)
        self.convl2 = TransformerConv(d_layer, d_layer, edge_dim = 10)
        # self.convl = torch.nn.Linear(args.d_graph_layer, 3)

        self.fc = nn.Linear(args.d_graph_layer+d_layer,1)
        self.relu = torch.nn.ReLU()
        self.elu = torch.nn.ELU()
        self.gelu = torch.nn.GELU()
        self.leakyrelu = torch.nn.LeakyReLU()

        self.Dropout = torch.nn.Dropout(p=args.dropout_rate)

        graph_pooling = args.graph_pooling
        
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "set2set":
            self.pool = Set2Set(args.d_graph_layer, processing_steps=2)
        if graph_pooling == "set2set":
            self.linl = nn.Linear((args.d_graph_layer+d_layer)*4, args.d_graph_layer+d_layer)
        else:
            self.linl = nn.Linear(args.d_graph_layer+d_layer, int((args.d_graph_layer+d_layer)))
            self.linl2 = nn.Linear(int((args.d_graph_layer+d_layer)), int(0.5*(args.d_graph_layer+d_layer)))
            self.fc = nn.Linear(int(0.5*(args.d_graph_layer+d_layer)),1)


        self.pool2 = global_mean_pool
        #self.pool2 = global_max_pool
        #self.pool2 = global_add_pool


    def forward(self, x, edge_index, edge_attr, batchs, x_apo, edge_index_apo, edge_attr_apo):
        x = self.conv1(x, edge_index, edge_attr)

        x = self.relu(x)
        x = self.Dropout(x)

        for i in range(len(self.convs)):
            #print('Layer:',i)

            x = self.convs[i](x, edge_index, edge_attr)
            # x = self.convs[i](x)
            x = self.relu(x)
            #x = self.bn(x)
            x = self.Dropout(x)


        x = self.pool(x,batchs)

        #apo#
        x_apo = self.conv2(x_apo, edge_index_apo, edge_attr_apo)

        x_apo = self.relu(x_apo)
        x_apo = self.Dropout(x_apo)

        for i in range(len(self.convs2)):
            #print('Layer:',i)

            x_apo = self.convs2[i](x_apo, edge_index_apo, edge_attr_apo)

            x_apo = self.relu(x_apo)
            #x = self.bn2(x_apo)

            x_apo = self.Dropout(x_apo)

        #x_apo = self.convl2(x_apo, edge_index_apo, edge_attr_apo)

        x_apo = self.pool2(x_apo, batchs)
        #x_apo = torch.mean(x_apo,dim=0,keepdim=True)

        #concat
        xc = torch.cat((x, x_apo), 1)
        #readout
        xc = self.linl(xc)
        xc = self.relu(xc)
        xc = self.Dropout(xc)
        xc = self.linl2(xc)
        xc = self.relu(xc)
        xc = self.Dropout(xc)         
        xc = self.fc(xc)
        return xc

class Net_holo(torch.nn.Module):
    def __init__(self, num_features, args):
        super().__init__()
        print("get the model of Net_holo")
        self.conv1 = TransformerConv(num_features, args.d_graph_layer, edge_dim = args.edge_dim)
        # self.conv1 = torch.nn.Linear(num_features, args.d_graph_layer)

        self.convs = torch.nn.ModuleList([TransformerConv(args.d_graph_layer, args.d_graph_layer, edge_dim = args.edge_dim) for _ in range(args.n_graph_layer)])
        # self.convs = torch.nn.ModuleList([torch.nn.Linear(args.d_graph_layer, args.d_graph_layer) for _ in range(args.n_graph_layer)])

        self.convl = TransformerConv(args.d_graph_layer, args.d_graph_layer, edge_dim = args.edge_dim)
        # self.convl = torch.nn.Linear(args.d_graph_layer, 3)

        self.fc = nn.Linear(args.d_graph_layer,1)
        self.relu = torch.nn.ReLU()
        self.elu = torch.nn.ELU()
        self.gelu = torch.nn.GELU()
        self.Dropout = torch.nn.Dropout(p=args.dropout_rate)

        graph_pooling = args.graph_pooling
        
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "set2set":
            self.pool = Set2Set(args.d_graph_layer, processing_steps=2)
        if graph_pooling == "set2set":
            self.linl = nn.Linear(2*args.d_graph_layer, args.d_graph_layer)
        else:
            self.linl = nn.Linear(args.d_graph_layer, args.d_graph_layer)

    def forward(self, x, edge_index, edge_attr, batchs):
        x = self.conv1(x, edge_index, edge_attr)
        # x = self.conv1(x)
        x = self.relu(x)
        x = self.Dropout(x)

        for i in range(len(self.convs)):
            #print('Layer:',i)

            x = self.convs[i](x, edge_index, edge_attr)
            # x = self.convs[i](x)
            x = self.relu(x)

            x = self.Dropout(x)

        x = self.convl(x, edge_index, edge_attr)


        x = self.pool(x,batchs)

        x = self.linl(x)
        x = self.relu(x)
        x = self.Dropout(x)        
        x = self.fc(x)

        return x