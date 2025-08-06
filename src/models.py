
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from dgl.nn import SAGEConv, GATConv, GraphConv, GINConv, ChebConv
from torch_geometric.utils import dropout_edge, negative_sampling, remove_self_loops, add_self_loops

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import ChebConv
import math
from torch.nn import Parameter

def cheby(i,x):
    if i==0:
        return 1
    elif i==1:
        return x
    else:
        T0=1
        T1=x
        for ii in range(2,i+1):
            T2=2*x*T1-T0
            T0,T1=T1,T2
        return T2

class ChebNetII(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k=3, dropout=0.5,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        ChebNetII: Adaptive Chebyshev Network with learnable coefficients and MLP head.

        Parameters:
        - in_feats: Dimension of input features
        - hidden_feats: Dimension of hidden features
        - out_feats: Number of output classes
        - k: Chebyshev polynomial order (K)
        - dropout: Dropout rate for regularization
        - device: Device to run the model on ('cpu' or 'cuda')
        """
        super(ChebNetII, self).__init__()
        self.k = k
        self.dropout = dropout
        self.device = device

        # Learnable Chebyshev coefficients
        self.temp = Parameter(torch.Tensor(k + 1))
        self.reset_parameters()

        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_feats, out_feats)
        ).to(device)

        # Batch Normalization
        self.norm = nn.BatchNorm1d(hidden_feats).to(device)

        # Dropout Layer
        self.dropout_layer = nn.Dropout(dropout)

    def reset_parameters(self):
        """
        Initialize Chebyshev coefficient parameters.
        """
        self.temp.data.fill_(1.0)

    def compute_adaptive_coefficients(self):
        """
        Compute adaptive Chebyshev coefficients using self.temp.
        Returns a tensor of size (k + 1,) on the specified device.
        """
        coe_tmp = F.relu(self.temp)
        coe = coe_tmp.clone()

        for i in range(self.k + 1):
            coe[i] = coe_tmp[0] * cheby(i, math.cos((self.k + 0.5) * math.pi / (self.k + 1)))
            for j in range(1, self.k + 1):
                x_j = math.cos((self.k - j + 0.5) * math.pi / (self.k + 1))
                coe[i] += coe_tmp[j] * cheby(i, x_j)
            coe[i] = 2 * coe[i] / (self.k + 1)

        coe[0] = coe[0] / 2  # Scale the first coefficient
        return coe.to(self.device)

    def forward(self, x_list, st=0, end=None):
        """
        Forward pass for ChebNetII.

        Parameters:
        - x_list: List of tensors [Tx_0, Tx_1, ..., Tx_K], each of shape (N, F)
        - st, end: Optional start/end indices for subsetting input (e.g., minibatching)

        Returns:
        - Log-softmax predictions of shape (N, C)
        """
        if end is None:
            end = x_list[0].size(0)

        # Compute Chebyshev coefficients
        coe = self.compute_adaptive_coefficients()

        # Weighted combination of Chebyshev basis features
        out = coe[0] * x_list[0][st:end, :].to(self.device)
        for k in range(1, self.k + 1):
            out += coe[k] * x_list[k][st:end, :].to(self.device)

        # Apply MLP head with normalization and dropout
        out = self.norm(out)
        out = self.dropout_layer(out)
        return F.log_softmax(self.mlp(out), dim=1)

class ACGNN_lamda_added(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k=3, dropout=0.3, epsilon=1e-4):
        """
        Efficient Graph Convolutional Network (EGCN) with:
        - Chebyshev Polynomial Approximation (Adaptive)
        - Early Stopping for Chebyshev Expansion
        - Three Aggregation Terms for Better Expressivity

        Parameters:
        - in_feats: Input feature size
        - hidden_feats: Hidden layer feature size
        - out_feats: Output feature size
        - k: Maximum Chebyshev polynomial order
        - dropout: Dropout rate for regularization
        - epsilon: Early stopping tolerance for Chebyshev computation
        """
        super(ACGNN, self).__init__()
        self.k = k  # Maximum Chebyshev order
        self.dropout = dropout
        self.epsilon = epsilon  # Stopping threshold

        # Chebyshev Convolution Layers
        self.cheb1 = ChebConv(in_feats, hidden_feats, k)
        self.cheb2 = ChebConv(hidden_feats, hidden_feats, k)
        self.cheb3 = ChebConv(hidden_feats, hidden_feats, k)  # Third aggregation term

        # Fully Connected Layer (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

        # Batch Normalization
        self.norm = nn.BatchNorm1d(hidden_feats)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, g, features):
        """
        Forward pass for EGCN with adaptive Chebyshev polynomial expansion.

        Parameters:
        - g: DGL graph
        - features: Input node features

        Returns:
        - Output predictions after passing through EGCN layers
        """
        # Compute lambda_max for Chebyshev polynomials
        lambda_max = dgl.laplacian_lambda_max(g)

        # First Chebyshev Convolution
        x = F.relu(self.cheb1(g, features, lambda_max=lambda_max))
        x = self.norm(x)

        # Early stopping check for Chebyshev polynomials
        prev_x = x.clone()
        for _ in range(1, self.k):
            x_new = F.relu(self.cheb2(g, x, lambda_max=lambda_max))
            if torch.norm(x_new - prev_x) < self.epsilon:
                break  # Stop if change is small
            prev_x = x_new.clone()
        
        # Third Aggregation Term for Feature Enhancement
        x_res = x  # Residual Connection
        x = F.relu(self.cheb3(g, x, lambda_max=lambda_max))
        x = self.dropout_layer(x) + x_res  # Efficient residual connection

        return self.mlp(x)

class ACGNN_ori(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k=2, dropout=0.3):
        """
        Speed-optimized Adaptive Chebyshev Graph Neural Network.
        
        Parameters:
        - in_feats: Input feature size
        - hidden_feats: Hidden layer feature size
        - out_feats: Output feature size
        - k: Chebyshev polynomial order (lower for speed)
        - dropout: Dropout rate
        """
        super(ACGNN, self).__init__()
        self.k = k  # Adaptive Chebyshev order
        self.dropout = dropout
        
        # Reduced ChebConv layers (only 2 instead of 3)
        self.cheb1 = ChebConv(in_feats, hidden_feats, k)
        self.cheb2 = ChebConv(hidden_feats, hidden_feats, k)
        
        # Fully Connected Layers
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )
        
        # Faster Normalization
        self.norm = nn.BatchNorm1d(hidden_feats)  # BatchNorm is faster than LayerNorm
        
        # Regularization
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, g, features):
        """
        Forward pass for Fast Adaptive ACGNN.
        
        Parameters:
        - g: DGL graph
        - features: Input node features
        
        Returns:
        - Output tensor after passing through Fast ACGNN layers
        """
        x = F.relu(self.cheb1(g, features))
        x = self.norm(x)  # BatchNorm improves stability
        
        x_res = x  # Residual Connection
        x = F.relu(self.cheb2(g, x))
        x = self.dropout_layer(x) + x_res  # Efficient Residual
        
        return self.mlp(x)

class ACGNN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k=3, dropout=0.3, epsilon=1e-4):
        """
        Efficient Graph Convolutional Network (EGCN) with:
        - Chebyshev Polynomial Approximation (Adaptive)
        - Early Stopping for Chebyshev Expansion
        - Three Aggregation Terms for Better Expressivity

        Parameters:
        - in_feats: Input feature size
        - hidden_feats: Hidden layer feature size
        - out_feats: Output feature size
        - k: Maximum Chebyshev polynomial order
        - dropout: Dropout rate for regularization
        - epsilon: Early stopping tolerance for Chebyshev computation
        """
        super(ACGNN, self).__init__()
        self.k = k  # Maximum Chebyshev order
        self.dropout = dropout
        self.epsilon = epsilon  # Stopping threshold

        # Chebyshev Convolution Layers
        self.cheb1 = ChebConv(in_feats, hidden_feats, k)
        self.cheb2 = ChebConv(hidden_feats, hidden_feats, k)
        self.cheb3 = ChebConv(hidden_feats, hidden_feats, k)  # Third aggregation term

        # Fully Connected Layer (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

        # Batch Normalization
        self.norm = nn.BatchNorm1d(hidden_feats)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, graph, features, lambda_max=None):
        if lambda_max is None:
            lambda_max = dgl.laplacian_lambda_max(graph)

        x = F.relu(self.cheb1(graph, features, lambda_max=lambda_max))
        x = self.norm(x)

        prev_x = x.clone()
        for _ in range(1, self.k):
            x_new = F.relu(self.cheb2(graph, x, lambda_max=lambda_max))
            if torch.norm(x_new - prev_x) < self.epsilon:
                break
            prev_x = x_new.clone()

        x_res = x
        x = F.relu(self.cheb3(graph, x, lambda_max=lambda_max))
        x = self.dropout_layer(x) + x_res

        return self.mlp(x)

class EGCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k=3, dropout=0.3, epsilon=1e-4):
        """
        Efficient Graph Convolutional Network (EGCN) with:
        - Chebyshev Polynomial Approximation (Adaptive)
        - Early Stopping for Chebyshev Expansion
        - Three Aggregation Terms for Better Expressivity

        Parameters:
        - in_feats: Input feature size
        - hidden_feats: Hidden layer feature size
        - out_feats: Output feature size
        - k: Maximum Chebyshev polynomial order
        - dropout: Dropout rate for regularization
        - epsilon: Early stopping tolerance for Chebyshev computation
        """
        super(ACGNN, self).__init__()
        self.k = k  # Maximum Chebyshev order
        self.dropout = dropout
        self.epsilon = epsilon  # Stopping threshold

        # Chebyshev Convolution Layers
        self.cheb1 = ChebConv(in_feats, hidden_feats, k)
        self.cheb2 = ChebConv(hidden_feats, hidden_feats, k)
        self.cheb3 = ChebConv(hidden_feats, hidden_feats, k)  # Third aggregation term

        # Fully Connected Layer (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

        # Batch Normalization
        self.norm = nn.BatchNorm1d(hidden_feats)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, graph, features, lambda_max=None):
        if lambda_max is None:
            lambda_max = dgl.laplacian_lambda_max(graph)

        x = F.relu(self.cheb1(graph, features, lambda_max=lambda_max))
        x = self.norm(x)

        prev_x = x.clone()
        for _ in range(1, self.k):
            x_new = F.relu(self.cheb2(graph, x, lambda_max=lambda_max))
            if torch.norm(x_new - prev_x) < self.epsilon:
                break
            prev_x = x_new.clone()

        x_res = x
        x = F.relu(self.cheb3(graph, x, lambda_max=lambda_max))
        x = self.dropout_layer(x) + x_res

        return self.mlp(x)

class ACGNN_ig(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k=3, dropout=0.3, epsilon=1e-4):
        super(ACGNN, self).__init__()
        self.k = k
        self.dropout = dropout
        self.epsilon = epsilon

        self.cheb1 = ChebConv(in_feats, hidden_feats, k)
        self.cheb2 = ChebConv(hidden_feats, hidden_feats, k)
        self.cheb3 = ChebConv(hidden_feats, hidden_feats, k)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

        self.norm = nn.BatchNorm1d(hidden_feats)
        self.dropout_layer = nn.Dropout(dropout)

        self.graph = None  # This will store the graph

    def set_graph(self, g):
        """Set the graph for forward pass — needed for Captum"""
        self.graph = g

    def forward(self, features):
        if self.graph is None:
            raise ValueError("Graph is not set. Use set_graph(g) before calling forward(features).")


        g = self.graph
        print("📦 Inside ACGNN forward:")
        print("   - features.shape:", features.shape)
        print("   - graph.num_nodes():", g.num_nodes())

        x = F.relu(self.cheb1(g, features))
        x = self.norm(x)

        prev_x = x.clone()
        for _ in range(1, self.k):
            x_new = F.relu(self.cheb2(g, x))
            if torch.norm(x_new - prev_x) < self.epsilon:
                break
            prev_x = x_new.clone()

        x_res = x
        x = F.relu(self.cheb3(g, x))
        x = self.dropout_layer(x) + x_res

        return self.mlp(x).squeeze()

class HGDC(torch.nn.Module):
    def __init__(self, args, weights=[0.95, 0.90, 0.15, 0.10]):
        super().__init__()
        self.args = args
        in_channels = self.args.in_channels
        hidden_channels = self.args.hidden_channels
        self.linear1 = Linear(in_channels, hidden_channels)

        # 3 convolutional layers for the original network
        self.conv_k1_1 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
        self.conv_k2_1 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops=False)
        self.conv_k3_1 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops=False)
        
        # 3 convolutional layers for the auxiliary network
        self.conv_k1_2 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
        self.conv_k2_2 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops=False)
        self.conv_k3_2 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops=False)

        self.linear_r0 = Linear(hidden_channels, 1)
        self.linear_r1 = Linear(2 * hidden_channels, 1)
        self.linear_r2 = Linear(2 * hidden_channels, 1)
        self.linear_r3 = Linear(2 * hidden_channels, 1)

        # Attention weights on outputs of different convolutional layers
        self.weight_r0 = torch.nn.Parameter(torch.Tensor([weights[0]]), requires_grad=True)
        self.weight_r1 = torch.nn.Parameter(torch.Tensor([weights[1]]), requires_grad=True)
        self.weight_r2 = torch.nn.Parameter(torch.Tensor([weights[2]]), requires_grad=True)
        self.weight_r3 = torch.nn.Parameter(torch.Tensor([weights[3]]), requires_grad=True)

    def forward(self, data):
        x_input = data.x
        edge_index_1 = data.edge_index
        edge_index_2 = data.edge_index_aux

        edge_index_1, _ = dropout_edge(edge_index_1, p=0.5, 
                                       force_undirected=True, 
                                       training=self.training)
        edge_index_2, _ = dropout_edge(edge_index_2, p=0.5, 
                                       force_undirected=True, 
                                       training=self.training)

        x_input = F.dropout(x_input, p=0.5, training=self.training)

        R0 = torch.relu(self.linear1(x_input))

        R_k1_1 = self.conv_k1_1(R0, edge_index_1)
        R_k1_2 = self.conv_k1_2(R0, edge_index_2)
        R1 = torch.cat((R_k1_1, R_k1_2), 1)

        R_k2_1 = self.conv_k2_1(R1, edge_index_1)
        R_k2_2 = self.conv_k2_2(R1, edge_index_2)
        R2 = torch.cat((R_k2_1, R_k2_2), 1)

        R_k3_1 = self.conv_k3_1(R2, edge_index_1)
        R_k3_2 = self.conv_k3_2(R2, edge_index_2)
        R3 = torch.cat((R_k3_1, R_k3_2), 1)

        R0 = F.dropout(R0, p=0.5, training=self.training)
        res0 = self.linear_r0(R0)
        R1 = F.dropout(R1, p=0.5, training=self.training)
        res1 = self.linear_r1(R1)
        R2 = F.dropout(R2, p=0.5, training=self.training)
        res2 = self.linear_r2(R2)
        R3 = F.dropout(R3, p=0.5, training=self.training)
        res3 = self.linear_r3(R3)

        out = res0 * self.weight_r0 + res1 * self.weight_r1 + res2 * self.weight_r2 + res3 * self.weight_r3
        return out

class MTGCN(torch.nn.Module):
    def __init__(self, args):
        super(MTGCN, self).__init__()
        self.args = args
        self.conv1 = ChebConv(58, 300, K=2, normalization="sym")
        self.conv2 = ChebConv(300, 100, K=2, normalization="sym")
        self.conv3 = ChebConv(100, 1, K=2, normalization="sym")

        self.lin1 = Linear(58, 100)
        self.lin2 = Linear(58, 100)

        self.c1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.c2 = torch.nn.Parameter(torch.Tensor([0.5]))

    def forward(self, data):
        edge_index, _ = dropout_edge(data.edge_index, p=0.5,
                                     force_undirected=True,
                                     num_nodes=data.x.size()[0],
                                     training=self.training)
        E = data.edge_index
        pb, _ = remove_self_loops(data.edge_index)
        pb, _ = add_self_loops(pb)

        x0 = F.dropout(data.x, training=self.training)
        x = torch.relu(self.conv1(x0, edge_index))
        x = F.dropout(x, training=self.training)
        x1 = torch.relu(self.conv2(x, edge_index))

        x = x1 + torch.relu(self.lin1(x0))
        z = x1 + torch.relu(self.lin2(x0))

        pos_loss = -torch.log(torch.sigmoid((z[E[0]] * z[E[1]]).sum(dim=1)) + 1e-15).mean()

        neg_edge_index = negative_sampling(pb, data.num_nodes, data.num_edges)

        neg_loss = -torch.log(
            1 - torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)) + 1e-15).mean()

        r_loss = pos_loss + neg_loss

        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return x, r_loss, self.c1, self.c2

class EMOGI(torch.nn.Module):
    def __init__(self,args):
        super(EMOGI, self).__init__()
        self.args = args
        self.conv1 = ChebConv(58, 300, K=2)
        self.conv2 = ChebConv(300, 100, K=2)
        self.conv3 = ChebConv(100, 1, K=2)

    def forward(self, data):
        edge_index = data.edge_index
        x = F.dropout(data.x, training=self.training)
        x = torch.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = torch.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return x

class ChebNet(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k=3):
        """
        ChebNet implementation using DGL's ChebConv.
        
        Parameters:
        - in_feats: Number of input features.
        - hidden_feats: Number of hidden layer features.
        - out_feats: Number of output features.
        - k: Chebyshev polynomial order.
        """
        super(ChebNet, self).__init__()
        self.cheb1 = ChebConv(in_feats, hidden_feats, k)
        self.cheb2 = ChebConv(hidden_feats, hidden_feats, k)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

    def forward(self, g, features):
        """
        Forward pass for ChebNet.
        
        Parameters:
        - g: DGL graph.
        - features: Input features tensor.
        
        Returns:
        - Output tensor after passing through ChebNet layers.
        """
        x = F.relu(self.cheb1(g, features))
        x = F.relu(self.cheb2(g, x))
        return self.mlp(x)

class GIN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GIN, self).__init__()
        # Define the first GIN layer
        self.gin1 = GINConv(
            nn.Sequential(
                nn.Linear(in_feats, hidden_feats),
                nn.ReLU(),
                nn.Linear(hidden_feats, hidden_feats)
            ),
            'mean'  # Aggregation method: 'mean', 'max', or 'sum'
        )
        # Define the second GIN layer
        self.gin2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_feats, hidden_feats),
                nn.ReLU(),
                nn.Linear(hidden_feats, hidden_feats)
            ),
            'mean'
        )
        # MLP for final predictions
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

    def forward(self, g, features):
        # Apply the first GIN layer
        x = F.relu(self.gin1(g, features))
        # Apply the second GIN layer
        x = F.relu(self.gin2(g, x))
        # Apply the MLP
        return self.mlp(x)

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GraphSAGE, self).__init__()
        self.sage1 = SAGEConv(in_feats, hidden_feats, aggregator_type='mean')
        self.sage2 = SAGEConv(hidden_feats, hidden_feats, aggregator_type='mean')
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

    def forward(self, g, features):
        x = F.relu(self.sage1(g, features))
        x = F.relu(self.sage2(g, x))
        return self.mlp(x)

class GAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads=3):
        """
        Graph Attention Network (GAT).
        
        Parameters:
        - in_feats: Number of input features.
        - hidden_feats: Number of hidden layer features.
        - out_feats: Number of output features.
        - num_heads: Number of attention heads.
        """
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_feats, hidden_feats, num_heads, activation=F.relu)
        self.gat2 = GATConv(hidden_feats * num_heads, hidden_feats, num_heads, activation=F.relu)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats * num_heads, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

    def forward(self, g, features):
        """
        Forward pass for GAT.
        
        Parameters:
        - g: DGL graph.
        - features: Input features tensor.
        
        Returns:
        - Output tensor after passing through GAT layers.
        """
        x = self.gat1(g, features)
        x = x.flatten(1)  # Flatten the output of multi-head attention
        x = self.gat2(g, x)
        x = x.flatten(1)  # Flatten the output again
        return self.mlp(x)

class GAT_relaevance(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads=3):
        """
        Graph Attention Network (GAT) with attention weight extraction.
        
        Parameters:
        - in_feats: Number of input features.
        - hidden_feats: Number of hidden layer features.
        - out_feats: Number of output features.
        - num_heads: Number of attention heads.
        """
        super(GAT, self).__init__()
        self.num_heads = num_heads
        self.gat1 = GATConv(in_feats, hidden_feats, num_heads, activation=F.relu)
        self.gat2 = GATConv(hidden_feats * num_heads, hidden_feats, num_heads, activation=F.relu)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats * num_heads, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

    def forward(self, g, features, return_attention=False):
        """
        Forward pass with optional attention weight return.

        Parameters:
        - g: DGL graph.
        - features: Node feature tensor.
        - return_attention: If True, returns attention weights.

        Returns:
        - Output tensor (and attention weights if requested).
        """
        if return_attention:
            x, attn1 = self.gat1(g, features, get_attention=True)
            x = x.flatten(1)
            x, attn2 = self.gat2(g, x, get_attention=True)
            x = x.flatten(1)
            out = self.mlp(x)
            return out, (attn1, attn2)
        else:
            x = self.gat1(g, features)
            x = x.flatten(1)
            x = self.gat2(g, x)
            x = x.flatten(1)
            return self.mlp(x)

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GCN, self).__init__()
        self.gcn1 = GraphConv(in_feats, hidden_feats)
        self.gcn2 = GraphConv(hidden_feats, hidden_feats)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

    def forward(self, g, features):
        x = F.relu(self.gcn1(g, features))
        x = F.relu(self.gcn2(g, x))
        return self.mlp(x)

class GIN_lrp_x(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, lrp_rule='epsilon'):
        super().__init__()
        self.lrp_rule = lrp_rule
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

        self.fc1 = nn.Linear(in_feats, hidden_feats)
        self.fc2 = nn.Linear(hidden_feats, hidden_feats)
        self.fc3 = nn.Linear(hidden_feats, hidden_feats)
        self.out_fc1 = nn.Linear(hidden_feats, hidden_feats)
        self.out_fc2 = nn.Linear(hidden_feats, out_feats)

        self.gin1 = GINConv(nn.Sequential(self.fc1, self.act1, self.fc2), aggregator_type='mean')
        self.gin2 = GINConv(nn.Sequential(self.fc3, self.act2), aggregator_type='mean')

        # Cache for forward pass
        self.cache = {}

    def forward(self, g, x):
        self.cache.clear()

        # Apply first GIN layer
        x_input = x.clone()
        x1 = self.fc1(x_input)
        self.cache['x1'] = x_input
        x = self.act1(x1)

        x2 = self.fc2(x)
        self.cache['x2'] = x2
        x = self.gin1(g, x2)  # Feed x2 to gin1, not reusing x2 again after gin1

        # Apply second GIN layer
        x3 = self.fc3(x)
        self.cache['x3'] = x3
        x = self.act2(x3)
        x = self.gin2(g, x)

        # Output MLP
        out1 = self.out_fc1(x)
        self.cache['out1'] = out1
        x = F.relu(out1)
        out2 = self.out_fc2(x)

        self.cache['x_out'] = x
        return out2


        def relprop(self, R, method='epsilon', epsilon=1e-6):
            """
            R: Relevance scores from output [batch_size, out_feats]
            Returns: Relevance scores per input feature [batch_size, in_feats]
            """
            if method == 'zplus':
                rule = lrp_linear_zplus
            else:
                rule = lambda i, w, o, r: lrp_linear_eps(i, w, o, r, epsilon=epsilon)

            # Output layer
            out_fc2_input = self.cache['x_out']
            R = rule(out_fc2_input, self.out_fc2.weight, None, R)

            # Output MLP
            R = rule(self.cache['out1'], self.out_fc1.weight, None, R)

            # GIN2 and act
            R = rule(self.cache['x3'], self.fc3.weight, None, R)

            # GIN1
            R = rule(self.cache['x2'], self.fc2.weight, None, R)

            # Input
            R = rule(self.cache['x1'], self.fc1.weight, None, R)

            return R

    def relprop(self, R, method='epsilon', epsilon=1e-6):
        """
        R: Relevance scores from output [batch_size, out_feats]
        Returns: Relevance scores per input feature [batch_size, in_feats]
        """
        if method == 'zplus':
            rule = lrp_linear_zplus
        else:
            rule = lambda i, w, o, r: lrp_linear_eps(i, w, o, r, epsilon=epsilon)

        # Output layer
        out_fc2_input = self.cache['x_out']
        R = rule(out_fc2_input, self.out_fc2.weight, None, R)

        # Output MLP
        R = rule(self.cache['out1'], self.out_fc1.weight, None, R)

        # GIN2 and act
        R = rule(self.cache['x3'], self.fc3.weight, None, R)

        # GIN1
        R = rule(self.cache['x2'], self.fc2.weight, None, R)

        # Input
        R = rule(self.cache['x1'], self.fc1.weight, None, R)

        return R

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ##bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        # Ensure targets are of type float
        targets = targets.float()

        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        probas = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probas, 1 - probas)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


#################################################
## _return_embeddings
#################################################


# class ACGNN(nn.Module):
#     def __init__(self, in_feats, hidden_feats, out_feats, k=3, dropout=0.3, epsilon=1e-4):
#         super(ACGNN, self).__init__()
#         self.k = k
#         self.dropout = dropout
#         self.epsilon = epsilon

#         self.cheb1 = ChebConv(in_feats, hidden_feats, k)
#         self.cheb2 = ChebConv(hidden_feats, hidden_feats, k)
#         self.cheb3 = ChebConv(hidden_feats, hidden_feats, k)

#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_feats, hidden_feats),
#             nn.ReLU(),
#             nn.Linear(hidden_feats, out_feats)
#         )

#         self.norm = nn.BatchNorm1d(hidden_feats)
#         self.dropout_layer = nn.Dropout(dropout)

#     def forward(self, graph, features, lambda_max=None, return_embeddings=False):
#         if lambda_max is None:
#             lambda_max = dgl.laplacian_lambda_max(graph)

#         x = F.relu(self.cheb1(graph, features, lambda_max=lambda_max))
#         x = self.norm(x)

#         prev_x = x.clone()
#         for _ in range(1, self.k):
#             x_new = F.relu(self.cheb2(graph, x, lambda_max=lambda_max))
#             if torch.norm(x_new - prev_x) < self.epsilon:
#                 break
#             prev_x = x_new.clone()

#         x_res = x
#         x = F.relu(self.cheb3(graph, x, lambda_max=lambda_max))
#         x = self.dropout_layer(x) + x_res

#         if return_embeddings:
#             return x  # Shape: [num_nodes, hidden_feats]

#         return self.mlp(x)

# class GIN(nn.Module):
#     def __init__(self, in_feats, hidden_feats, out_feats):
#         super(GIN, self).__init__()
#         # Define the first GIN layer
#         self.gin1 = GINConv(
#             nn.Sequential(
#                 nn.Linear(in_feats, hidden_feats),
#                 nn.ReLU(),
#                 nn.Linear(hidden_feats, hidden_feats)
#             ),
#             'mean'
#         )
#         # Define the second GIN layer
#         self.gin2 = GINConv(
#             nn.Sequential(
#                 nn.Linear(hidden_feats, hidden_feats),
#                 nn.ReLU(),
#                 nn.Linear(hidden_feats, hidden_feats)
#             ),
#             'mean'
#         )
#         # MLP for final predictions
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_feats, hidden_feats),
#             nn.ReLU(),
#             nn.Linear(hidden_feats, out_feats)
#         )

#     def forward(self, g, features, return_embeddings=False):
#         # Apply the first and second GIN layers
#         x = F.relu(self.gin1(g, features))
#         x = F.relu(self.gin2(g, x))

#         if return_embeddings:
#             return x  # shape: [num_nodes, hidden_feats]

#         return self.mlp(x)

# class GCN(nn.Module):
#     def __init__(self, in_feats, hidden_feats, out_feats):
#         super(GCN, self).__init__()
#         self.gcn1 = GraphConv(in_feats, hidden_feats)
#         self.gcn2 = GraphConv(hidden_feats, hidden_feats)
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_feats, hidden_feats),
#             nn.ReLU(),
#             nn.Linear(hidden_feats, out_feats)
#         )

#     def forward(self, g, features, return_embeddings=False):
#         x = F.relu(self.gcn1(g, features))
#         x = F.relu(self.gcn2(g, x))

#         if return_embeddings:
#             return x  # shape: [num_nodes, hidden_feats]

#         return self.mlp(x)

# class GraphSAGE(nn.Module):
#     def __init__(self, in_feats, hidden_feats, out_feats):
#         super(GraphSAGE, self).__init__()
#         self.sage1 = SAGEConv(in_feats, hidden_feats, aggregator_type='mean')
#         self.sage2 = SAGEConv(hidden_feats, hidden_feats, aggregator_type='mean')
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_feats, hidden_feats),
#             nn.ReLU(),
#             nn.Linear(hidden_feats, out_feats)
#         )

#     def forward(self, g, features, return_embeddings=False):
#         x = F.relu(self.sage1(g, features))
#         x = F.relu(self.sage2(g, x))

#         if return_embeddings:
#             return x  # shape: [num_nodes, hidden_feats]

#         return self.mlp(x)

# class GAT(nn.Module):
#     def __init__(self, in_feats, hidden_feats, out_feats, num_heads=3):
#         """
#         Graph Attention Network (GAT).

#         Parameters:
#         - in_feats: Number of input features.
#         - hidden_feats: Number of hidden layer features.
#         - out_feats: Number of output features.
#         - num_heads: Number of attention heads.
#         """
#         super(GAT, self).__init__()
#         self.gat1 = GATConv(in_feats, hidden_feats, num_heads, activation=F.relu)
#         self.gat2 = GATConv(hidden_feats * num_heads, hidden_feats, num_heads, activation=F.relu)
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_feats * num_heads, hidden_feats),
#             nn.ReLU(),
#             nn.Linear(hidden_feats, out_feats)
#         )

#     def forward(self, g, features, return_embeddings=False):
#         """
#         Forward pass for GAT.

#         Parameters:
#         - g: DGL graph.
#         - features: Input features tensor.
#         - return_embeddings: If True, return node embeddings before MLP.

#         Returns:
#         - Either the output tensor after the MLP or intermediate node embeddings.
#         """
#         x = self.gat1(g, features)
#         x = x.flatten(1)  # Flatten the output of multi-head attention
#         x = self.gat2(g, x)
#         x = x.flatten(1)  # Flatten the output again

#         if return_embeddings:
#             return x  # Shape: [num_nodes, hidden_feats * num_heads]

#         return self.mlp(x)

# class ChebNet(nn.Module):
#     def __init__(self, in_feats, hidden_feats, out_feats, k=3):
#         """
#         ChebNet implementation using DGL's ChebConv.

#         Parameters:
#         - in_feats: Number of input features.
#         - hidden_feats: Number of hidden layer features.
#         - out_feats: Number of output features.
#         - k: Chebyshev polynomial order.
#         """
#         super(ChebNet, self).__init__()
#         self.cheb1 = ChebConv(in_feats, hidden_feats, k)
#         self.cheb2 = ChebConv(hidden_feats, hidden_feats, k)
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_feats, hidden_feats),
#             nn.ReLU(),
#             nn.Linear(hidden_feats, out_feats)
#         )

#     def forward(self, g, features, return_embeddings=False):
#         """
#         Forward pass for ChebNet.

#         Parameters:
#         - g: DGL graph.
#         - features: Input features tensor.
#         - return_embeddings: If True, return node embeddings before MLP.

#         Returns:
#         - Either the output tensor after the MLP or intermediate node embeddings.
#         """
#         x = F.relu(self.cheb1(g, features))
#         x = F.relu(self.cheb2(g, x))

#         if return_embeddings:
#             return x  # Shape: [num_nodes, hidden_feats]

#         return self.mlp(x)
    