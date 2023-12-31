import torch as th
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from scipy.sparse import csr_matrix
import copy
from torch.nn import Parameter

        
        
class MRGNN(nn.Module):
    def __init__(self, in_dims, h_dim, out_dim, num_classes, num_nodes, n_mlp_layer, threshold, threshold_c, hm_dim, dropout=0, transf_type=2, triples =None, alpha=10, beta=1,):
        super(MRGNN, self).__init__()
        self.gene_emb = Parameter(torch.randn(4264, 4264))
        self.in_dims = in_dims
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.triples = triples
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.threshold_c = threshold_c
        self.hm_dim = hm_dim
        self.transf_type = transf_type
        self.node_embeddings = nn.Parameter(torch.FloatTensor(num_nodes, self.h_dim))
        nn.init.kaiming_normal_(self.node_embeddings, mode='fan_in')
        self.linear = nn.Linear(self.h_dim, self.h_dim)
        self.out_layer = OneLayerNet(self.hm_dim, num_classes, self.dropout)

        self.build_model()


        self.type_norm='pair'
        if self.type_norm in ['None', 'batch', 'pair']:
            skip_connect = False
        else:
            skip_connect = True
        self.bat=batch_norm(32, self.type_norm, skip_connect, 10, 0.005)

        self.relu=nn.ReLU(True)
    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        self.i2h = self.build_input_layer()

    def forward(self, x , num_nodes, num_relations):
        device = 'cuda'
        features_list = []
        features_list.append(x)
        num_nodes = num_nodes
        num_relations = num_relations
        vertical_stacking= True
        threshold_c = self.threshold_c
        threshold = self.threshold
        alpha = self.alpha
        beta = self.beta
        in_dim = self.hm_dim
        triples = self.triples
        transf_type = self.transf_type
        features_list[0] = features_list[0][:,1613:]
        features_list.append(self.gene_emb.to(device))

        #Feature initialization and transformation
        def transfer(transf_type):
            if transf_type==0:
                h =self.node_embeddings
                h = self.linear(h)
            elif transf_type==1:
                h = []
                for i2h, feature in zip(self.i2h, features_list):
                    h.append(i2h(feature))
                h = th.cat(h, 0)
            elif transf_type==2:
                h = []
                for i2h, feature in zip(self.i2h, features_list):

                    h.append(i2h(feature))
                h = th.cat(h, 0)

            return h
        h = transfer(transf_type)
        
        
        with torch.no_grad():
            inverse_triples = generate_inverses(triples, num_relations)
            triples_plus = torch.cat([triples, inverse_triples], dim=0)

        with torch.no_grad():
            # Stack adjacency matrices either vertically or horizontally
            adj_indices, adj_size = stack_matrices(
                triples_plus,
                num_nodes,
                num_relations,
                vertical_stacking=vertical_stacking,
                device=device
            )

            num_triples = adj_indices.size(0)
            vals = torch.ones(num_triples, dtype=torch.float, device=device)
            
            # Apply normalisation (vertical-stacking -> row-wise rum & horizontal-stacking -> column-wise sum)
            sums = sum_sparse(adj_indices, vals, adj_size, row_normalisation=vertical_stacking, device=device)
            vals_or = vals
            vals = vals / sums

            if device == 'cuda':
                adj = torch.cuda.sparse.FloatTensor(indices=adj_indices.t(), values=vals, size=adj_size)
            else:
                adj = torch.sparse.FloatTensor(indices=adj_indices.t(), values=vals, size=adj_size)


            laplace_r = []  # Storage Laplace for R ralations
            adj_r = []
            adj_csr = csr_matrix((vals_or.cpu(), adj_indices.t().cpu()), shape=adj_size) 
            adj_inter = csr_matrix((vals.cpu(), adj_indices.t().cpu()), shape=adj_size)

            for r in range(0, (num_relations)):
                v1 = r * num_nodes
                b = adj_csr[v1:(v1 + num_nodes), :]
                c = adj_inter[v1:(v1 + num_nodes), :]
                if r < (num_relations):
                    norml, normA = calc_L_hat(b)
                else:
                    norml, normA = calc_Lself_hat(b)  # selfloop adj
                J = sparse_mx_to_torch_sparse_tensor(norml)
                A = sparse_mx_to_torch_sparse_tensor(c)
                L = J.float()
                laplace_r.append(L)
                adj_r.append(A)

        # Initialization coefficients and feature
        u = torch.full((num_relations, 1), 1 / (num_relations)).to(device)
        output_ = h.to(device)
        means = output_.mean(1, keepdim=True)
        deviations = output_.std(1, keepdim=True)
        output_ = (output_ - means) / deviations
        output_ = torch.where(torch.isnan(output_), torch.full_like(output_, 0), output_)
        output = output_.clone()
        # Calculating the initial value of totalvariation
        w = calculate_relational_totalvariation(num_relations, adj_r, laplace_r, output, num_nodes)
        c_l1tr = torch.norm(w, p=1, dim=0).to(device)  # l1 norm for totalvariation vector

        H_=output_
        #Message passing Layer
        while c_l1tr >= 0:
           # Updating Coefficient, EMDA Algorithm 1 in paper
            w = calculate_relational_totalvariation(num_relations, adj_r, laplace_r, output, num_nodes)
            l1tr = torch.norm(w, p=1, dim=0).to(device)  # l1 norm for totalvariation vector
            fi = l1tr + ((2 * beta) / alpha)  # Lipschitz constant
            condition = 0
            t = 11
            while condition >= 0:
                u_before = copy.deepcopy(u)
                T_t = ((2 * math.log(num_relations)) / (t * (fi * fi))).sqrt()  # var:t
                f_de = (((2 * beta) / alpha) * u.to(device)) + w.to(device)  # var:u
                u_ta = torch.mul(u.to(device), torch.exp(-T_t * f_de).to(device))  # var:u,t
                t = t + 1
                for r in range(num_relations):
                    u_tamp = u_ta[r]  
                    u_tampm = torch.sum(u_ta)
                    u_tnext = u_tamp / u_tampm
                    u[r] = u_tnext
                condition = torch.sqrt(torch.sum(torch.square(u_before - u)))
                if condition.item() < threshold_c:
                    break
            condition_layer = l1tr / c_l1tr
            if condition_layer.item() < threshold:
                break

            # Updating Features
            H = output_
            u1 = copy.deepcopy(u)
            u1 = u1.reshape(num_relations, 1, 1)
            af = torch.spmm(adj.data.to(device), output.to(device))
            af1 = af.reshape(num_relations, num_nodes, in_dim)
            afw = torch.einsum('rno, rii -> no', af1.to(device), u1.to(device))
            output = (1 / (1 + alpha)) * H.to(device) + (alpha / (1 + alpha)) * afw.to(device)
            output = output.float().to(device)

        embeddings = output
        output=self.bat(output)
        output = self.out_layer(output)
        
        return output, embeddings
