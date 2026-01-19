from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, GCNConv, global_mean_pool
import torch
import torch.nn.functional as F
import torch.nn as nn

class HGOOD(nn.Module):

    def __init__(self, hidden_dim, num_gc_layers, num_hp_layers, feat_dim, str_dim):
        super(HCL, self).__init__()  

        self.embedding_dim = hidden_dim * num_gc_layers 
        self.hidden_dim = hidden_dim  
        self.num_gc_layers = num_gc_layers  
        self.encoder_feat = Encoder_GIN(feat_dim, hidden_dim, num_gc_layers)  
        self.encoder_str = Encoder_GIN(str_dim, hidden_dim, num_gc_layers)  
        self.encoder_hyper_feat = HypergraphEncoder(feat_dim, hidden_dim, num_hp_layers)  
        self.encoder_hyper_str = HypergraphEncoder(str_dim, hidden_dim, num_hp_layers) 
        
        self.proj_head_feat_g = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),  
            nn.ReLU(inplace=True), 
            nn.Linear(self.embedding_dim, self.embedding_dim)  
        )

        self.proj_head_str_g = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

        self.proj_head_feat_n = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

        self.proj_head_str_n = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

        self.proj_head_b = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
        self.proj_head_hyper_feat_g = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

        self.proj_head_hyper_str_g = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

        self.proj_head_hyper_feat_n = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

        self.proj_head_hyper_str_n = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )


        self.proj_head_hyper_b = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

        self.init_emb()  

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim  
        for m in self.modules():  
            if isinstance(m, nn.Linear):  
                torch.nn.init.xavier_uniform_(m.weight.data) 
                if m.bias is not None:  
                    m.bias.data.fill_(0.0) 

    def get_b(self, x_f, x_s, edge_index, batch, num_graphs):
        g_f, _ = self.encoder_feat(x_f, edge_index, batch)  
        g_s, _ = self.encoder_str(x_s, edge_index, batch) 
        b = self.proj_head_b(torch.cat((g_f, g_s), 1)) 
        return b

    def forward(self, x_f, x_s, edge_index, batch, num_graphs):
        g_f, n_f = self.encoder_feat(x_f, edge_index, batch)  
        g_s, n_s = self.encoder_str(x_s, edge_index, batch) 
        g_hyper_f, n_hyper_f = self.encoder_hyper_feat(x_f, batch, num_graphs) 
        g_hyper_s, n_hyper_s = self.encoder_hyper_str(x_s, batch, num_graphs) 

        b = self.proj_head_b(torch.cat((g_f, g_s), 1))  
        hyper_b = self.proj_head_hyper_b(torch.cat((g_hyper_f, g_hyper_s), 1))  

        g_f = self.proj_head_feat_g(g_f) 
        g_s = self.proj_head_str_g(g_s) 
        n_f = self.proj_head_feat_n(n_f)
        n_s = self.proj_head_str_n(n_s)  
        
        g_hyper_f = self.proj_head_hyper_feat_g(g_hyper_f)  
        g_hyper_s = self.proj_head_hyper_str_g(g_hyper_s) 
        n_hyper_f = self.proj_head_hyper_feat_n(n_hyper_f)  
        n_hyper_s = self.proj_head_hyper_str_n(n_hyper_s) 
        return b, hyper_b, g_f, g_s, n_f, n_s, g_hyper_f, g_hyper_s, n_hyper_f, n_hyper_s

    @staticmethod
    def calc_cross_modal_prototype_loss(instances, cross_modal_prototypes, im2cluster, temperature=0.2):
        batch_size, _ = instances.size() 
        instances = F.normalize(instances, dim=1)  
        cross_modal_prototypes = F.normalize(cross_modal_prototypes, dim=1)        
        sim_matrix = torch.matmul(instances, cross_modal_prototypes.t()) 
        sim_matrix = torch.exp(sim_matrix / temperature)        
        pos_proto_id = im2cluster 
        pos_sim = sim_matrix[range(batch_size), pos_proto_id] 
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)  
        loss = - torch.log(loss + 1e-12)         
        return loss

    @staticmethod
    def calc_bidirectional_cross_modal_loss(b, hyper_b, cluster_result_b, cluster_result_hyper, temperature=0.2):
        im2cluster_b, prototypes_b = cluster_result_b['im2cluster'], cluster_result_b['centroids'] 
        im2cluster_hyper, prototypes_hyper = cluster_result_hyper['im2cluster'], cluster_result_hyper['centroids']        
        loss_b_to_hyper = HCL.calc_cross_modal_prototype_loss(b, prototypes_hyper, im2cluster_b, temperature)
        loss_hyper_to_b = HCL.calc_cross_modal_prototype_loss(hyper_b, prototypes_b, im2cluster_hyper, temperature)     
        cross_modal_loss = loss_b_to_hyper + loss_hyper_to_b        
        return cross_modal_loss

    @staticmethod
    def scoring_cross_modal(instances, cross_modal_prototypes, temperature=0.2):
        instances = F.normalize(instances, dim=1)  
        cross_modal_prototypes = F.normalize(cross_modal_prototypes, dim=1)  
        sim_matrix = torch.matmul(instances, cross_modal_prototypes.t())  
        sim_matrix = torch.exp(sim_matrix / temperature)  
        min_sim, _ = torch.min(sim_matrix, dim=1)       
        return min_sim 

    @staticmethod
    def scoring_b(b, cluster_result, temperature=0.2):
        im2cluster, prototypes, density = cluster_result['im2cluster'], cluster_result['centroids'], cluster_result['density']  
        batch_size, _ = b.size() 
        b_abs = b.norm(dim=1) 
        prototypes_abs = prototypes.norm(dim=1)      
        sim_matrix = torch.einsum('ik,jk->ij', b, prototypes) / torch.einsum('i,j->ij', b_abs, prototypes_abs)
        sim_matrix = torch.exp(sim_matrix / (temperature * density)) 
        v, id = torch.min(sim_matrix, 1)  
        return v  

    @staticmethod
    def calc_loss_b(b, index, cluster_result, temperature=0.2):
        im2cluster, prototypes, density = cluster_result['im2cluster'], cluster_result['centroids'], cluster_result['density']  
        pos_proto_id = im2cluster[index].cpu().tolist()  
        batch_size, _ = b.size()  
        b_abs = b.norm(dim=1)  
        prototypes_abs = prototypes.norm(dim=1)  
        sim_matrix = torch.einsum('ik,jk->ij', b, prototypes) / torch.einsum('i,j->ij', b_abs, prototypes_abs)  # 余弦相似度
        sim_matrix = torch.exp(sim_matrix / (temperature * density))  
        pos_sim = sim_matrix[range(batch_size), pos_proto_id]  

        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)  
        loss = - torch.log(loss + 1e-12)  
        return loss  

    @staticmethod
    def calc_loss_n(x, x_aug, batch, temperature=0.2):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1) 
        x_aug_abs = x_aug.norm(dim=1)  

        node_belonging_mask = batch.repeat(batch_size, 1) 
        node_belonging_mask = node_belonging_mask == node_belonging_mask.t()  
        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)  
        sim_matrix = torch.exp(sim_matrix / temperature) * node_belonging_mask  
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]  

        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim + 1e-12) 
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim + 1e-12)  
        loss_0 = - torch.log(loss_0)  
        loss_1 = - torch.log(loss_1) 
        loss = (loss_0 + loss_1) / 2.0  
        loss = global_mean_pool(loss, batch)  
        return loss  

    @staticmethod
    def calc_loss_g(x, x_aug, temperature=0.2):
        batch_size, _ = x.size()  
        x_abs = x.norm(dim=1)  
        x_aug_abs = x_aug.norm(dim=1) 

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)  
        sim_matrix = torch.exp(sim_matrix / temperature)  
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]  

        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim) 
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss_0 = - torch.log(loss_0)  
        loss_1 = - torch.log(loss_1)  
        loss = (loss_0 + loss_1) / 2.0 
        return loss  



class HypergraphEncoder(torch.nn.Module):

    def __init__(self, num_features, dim, num_gc_layers, num_hyperedges=16):
        super(HypergraphEncoder, self).__init__()  
        self.num_gc_layers = num_gc_layers  
        self.dim = dim  
        self.num_hyperedges = num_hyperedges  
        self.embedding_dim = dim * num_gc_layers  
        self.hyperedge_embeddings = nn.Parameter(torch.Tensor(num_hyperedges, dim))  
        nn.init.xavier_uniform_(self.hyperedge_embeddings)   
        self.hyper_convs = torch.nn.ModuleList()  
        for i in range(num_gc_layers):
            if i:
                nn_layer = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))  
            else:
                nn_layer = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))  
            conv = HypergraphConv(nn_layer)  
            self.hyper_convs.append(conv) 

    def forward(self, x, batch, num_graphs):
        batch_size = num_graphs  
        hyper_adj_list = self.build_hypergraph_adjacency(x, batch, batch_size)  
        xs = [] 
        current_x = x  
        
        for i in range(self.num_gc_layers):
            conv_results = [] 
            current_idx = 0 
            new_x = torch.zeros(x.size(0), self.dim, device=x.device)           
            for graph_idx in range(batch_size):
                graph_mask = (batch == graph_idx)  
                num_nodes = graph_mask.sum().item()                
                if num_nodes == 0:  
                    continue                   
                graph_x = current_x[graph_mask]  

                if graph_idx < len(hyper_adj_list):  
                    hyper_adj = hyper_adj_list[graph_idx]  
                    
                    if hyper_adj.size(0) == num_nodes and hyper_adj.size(1) == self.num_hyperedges:
                        graph_x_conv = self.hyper_convs[i](graph_x, hyper_adj)
                        new_x[graph_mask] = graph_x_conv 
                
                current_idx += num_nodes  
            
            current_x = F.relu(new_x)  
            xs.append(current_x)  
        
        xpool = [global_add_pool(x, batch) for x in xs]  
        graph_embeddings = torch.cat(xpool, 1)     
        node_embeddings = torch.cat(xs, 1)   
        return graph_embeddings, node_embeddings 

    def build_hypergraph_adjacency(self, x, batch, batch_size):
        hyper_adj_list = []         
        for i in range(batch_size):
            graph_mask = (batch == i)  
            graph_nodes = x[graph_mask]  
            num_nodes = graph_mask.sum().item() 
            
            if num_nodes == 0:
                continue
                
            node_embeddings = graph_nodes 
            hyperedge_emb = self.hyperedge_embeddings  
            
            if node_embeddings.size(1) != hyperedge_emb.size(1):
                if not hasattr(self, 'dim_adjust'): 
                    self.dim_adjust = Linear(node_embeddings.size(1), hyperedge_emb.size(1)).to(x.device)  
                node_embeddings = self.dim_adjust(node_embeddings)  
            
            similarity = torch.matmul(node_embeddings, hyperedge_emb.t())  
            hyper_adj = F.softmax(similarity, dim=1) 
            
            hyper_adj_list.append(hyper_adj)       
        return hyper_adj_list 

class HypergraphConv(nn.Module):
    def __init__(self, nn_module):
        super(HypergraphConv, self).__init__()  
        self.nn_module = nn_module  

    def forward(self, x, hyper_adj):
        latent_embeddings = torch.matmul(hyper_adj.t(), x)  
        propagated_embeddings = torch.matmul(hyper_adj, latent_embeddings)  
        output = self.nn_module(propagated_embeddings)         
        return output 

class Encoder_GIN(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Encoder_GIN, self).__init__()  
        self.num_gc_layers = num_gc_layers 
        self.convs = torch.nn.ModuleList()  

        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim)) 
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))  
            conv = GINConv(nn)  
            self.convs.append(conv)  

    def forward(self, x, edge_index, batch):
        xs = [] 
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index)) 
            xs.append(x)  

        xpool = [global_add_pool(x, batch) for x in xs]  
        x = torch.cat(xpool, 1)  
        return x, torch.cat(xs, 1)  

class Encoder_GCN(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Encoder_GCN, self).__init__() 
        self.num_gc_layers = num_gc_layers  
        self.convs = torch.nn.ModuleList() 

        for i in range(num_gc_layers):
            if i:
                conv = GCNConv(dim, dim)  
            else:
                conv = GCNConv(num_features, dim)  
            self.convs.append(conv) 

    def forward(self, x, edge_index, batch):
        xs = []  
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))  
            xs.append(x)  

        xpool = [global_mean_pool(x, batch) for x in xs] 
        x = torch.cat(xpool, 1)  
        return x, torch.cat(xs, 1)  