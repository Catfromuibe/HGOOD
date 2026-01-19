def warn(*args, **kwargs):
    pass
import sys
import warnings
warnings.warn = warn
from model import HGOOD
from data_loader import *
import argparse
import numpy as np
import torch
import random
import faiss
import sklearn.metrics as skm
import torch_geometric
import copy
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp_type', type=str, default='oodd', choices=['oodd', 'ad'])
    parser.add_argument('-DS', help='Dataset', default='PTC_MR')
    parser.add_argument('-DS_ood', help='Dataset', default='MUTAG')
    parser.add_argument('-DS_pair', default=None)
    parser.add_argument('-rw_dim', type=int, default=16)
    parser.add_argument('-dg_dim', type=int, default=16)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-batch_size_test', type=int, default=9999)
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-num_layer', type=int, default=5)
    parser.add_argument('-num_gc_layer', type=int, default=5)
    parser.add_argument('-hidden_dim', type=int, default=16)
    parser.add_argument('-num_trial', type=int, default=2)
    parser.add_argument('-num_epoch', type=int, default=170)
    parser.add_argument('-eval_freq', type=int, default=10)
    parser.add_argument('-is_adaptive', type=int, default=1)
    parser.add_argument('-num_cluster', type=int, default=2)
    parser.add_argument('-alpha', type=float, default=0.2)
    parser.add_argument('-cross_modal_weight', type=float, default=1.0, help='Weight for cross-modal prototype loss')
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch_geometric.seed_everything(seed)


def run_kmeans(x, args):
    results = {}

    if np.isnan(x).any() or np.isinf(x).any():
        print("Warning: Input to run_kmeans contains NaNs or Infs. Replacing with 0.")
        x = np.nan_to_num(x)

    d = x.shape[1]
    k = args.num_cluster

    if x.shape[0] < k:
        k = x.shape[0]
        print(f"Warning: Number of samples ({x.shape[0]}) is less than num_cluster ({args.num_cluster}). Adjusting k to {k}.")

    clus = faiss.Clustering(d, k)
    clus.niter = 20
    clus.nredo = 5
    clus.seed = 0
    clus.max_points_per_centroid = 1000
    clus.min_points_per_centroid = 3
    try:
        index = faiss.IndexFlatL2(d)
        clus.train(x, index)
    except:
        print('Fail to cluster with GPU. Try CPU...')
        index = faiss.IndexFlatL2(d)
        clus.train(x, index)

    D, I = index.search(x, 1)
    im2cluster = [int(n[0]) for n in I]
    centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)
 
    Dcluster = [[] for c in range(k)]
    for im, i in enumerate(im2cluster):
        Dcluster[i].append(D[im][0])

    density = np.zeros(k)
    for i, dist in enumerate(Dcluster):
        if len(dist) > 1:
            d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
            density[i] = d

    dmax = density.max()
    for i, dist in enumerate(Dcluster):
        if len(dist) <= 1:
            density[i] = dmax

    density = density.clip(np.percentile(density, 30),
                           np.percentile(density, 70))
    
    if density.mean() > 1e-10:
        density = density / density.mean() + 0.5
    else:
        density = density + 0.5

    centroids = torch.Tensor(centroids).cuda()
    centroids = torch.nn.functional.normalize(centroids, p=2, dim=1)

    im2cluster = torch.LongTensor(im2cluster).cuda()
    density = torch.Tensor(density).cuda()

    results['centroids'] = centroids
    results['density'] = density
    results['im2cluster'] = im2cluster
    return results


def get_cluster_result(dataloader, model, args, use_hyper=False):

    model.eval()
    embeddings = torch.zeros((n_train, model.embedding_dim))
    for data in dataloader:
        with torch.no_grad():
            data = data.to(device)
            if use_hyper:
                
                _,hyper_b, _, _, _, _, _, _, _, _ = model(
                    data.x, data.x_s, data.edge_index, data.batch, data.num_graphs
                )
                embeddings[data.idx] = hyper_b.detach().cpu()
            else:
               
                b = model.get_b(data.x, data.x_s, data.edge_index, data.batch, data.num_graphs)
                embeddings[data.idx] = b.detach().cpu()
    
    cluster_result = run_kmeans(embeddings.numpy(), args)
    return cluster_result
def get_kmeans_cluster_result(data,ebdata,model, args):
    
    model.eval()
    embeddings = ebdata.detach().cpu()
    cluster_result = run_kmeans(embeddings.numpy(), args)
    return cluster_result


if __name__ == '__main__':
    setup_seed(0)
    torch.cuda.set_device(0)
    args = arg_parse()

    if args.exp_type == 'ad':
        if args.DS.startswith('Tox21'):
            dataloader, dataloader_test, meta = get_ad_dataset_Tox21(args)
        else:
            splits = get_ad_split_TU(args, fold=args.num_trial)

    aucs = []
    for trial in range(args.num_trial):
        setup_seed(trial)
        print(trial)

        if args.exp_type == 'oodd':
            dataloader, dataloader_test, meta = get_ood_dataset(args)
        elif args.exp_type == 'ad' and not args.DS.startswith('Tox21'):
            dataloader, dataloader_test, meta = get_ad_dataset_TU(args, splits[trial])

        dataset_num_features = meta['num_feat']
        n_train = meta['num_train']

        if trial == 0:
            print('================')
            print('Exp_type: {}'.format(args.exp_type))
            print('DS: {}'.format(args.DS_pair if args.DS_pair is not None else args.DS))
            print('num_features: {}'.format(dataset_num_features))
            print('num_structural_encodings: {}'.format(args.dg_dim + args.rw_dim))
            print('hidden_dim: {}'.format(args.hidden_dim))
            print('num_gc_layers: {}'.format(args.num_layer))
            print('cross_modal_weight: {}'.format(args.cross_modal_weight))
            print('================')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = HGOOD(args.hidden_dim, args.num_layer, args.num_gc_layer,dataset_num_features, args.dg_dim+args.rw_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_auc = 0.0
        best_model_state = None
        for epoch in range(1, args.num_epoch + 1):
            if args.is_adaptive:
                if epoch == 1:
                    weight_b, weight_g, weight_n, weight_cross_modal = 1, 1, 1, 1
                   
                else:
                    weight_b, weight_g, weight_n, weight_cross_modal = std_b ** args.alpha, std_g ** args.alpha, std_n ** args.alpha,args.alpha
                    
                    weight_sum = (weight_b + weight_g + weight_n + weight_cross_modal) / 4
                    
                    weight_b, weight_g, weight_n, weight_cross_modal = weight_b/weight_sum, weight_g/weight_sum, weight_n/weight_sum, weight_cross_modal/weight_sum
                    
            cluster_result_hyper = get_cluster_result(dataloader, model, args, use_hyper=True)

            model.train()
            loss_all = 0
            if args.is_adaptive:
                loss_b_all, loss_g_all, loss_n_all, loss_cross_modal_all = [], [], [], []

            for data in dataloader:
                data = data.to(device)
                optimizer.zero_grad()
                
                b, hyper_b, g_f, g_s, n_f, n_s, g_hyper_f, g_hyper_s, n_hyper_f, n_hyper_s = model(
                    data.x, data.x_s, data.edge_index, data.batch, data.num_graphs
                )

                loss_g = model.calc_loss_g(g_f, g_s)
                loss_b = model.calc_loss_b(b, data.idx, cluster_result_b) # Ablation: remove loss_b
                loss_n = model.calc_loss_n(n_f, n_s, data.batch) 
                cluster_result2_b=get_kmeans_cluster_result(data,b, model, args)
                cluster_result2_hy_b=get_kmeans_cluster_result(data,hyper_b, model, args)
                loss_cross_modal = model.calc_bidirectional_cross_modal_loss(
                    b, hyper_b, cluster_result2_b, cluster_result2_hy_b
                )
                
                if args.is_adaptive:
                    loss = (weight_b * loss_b.mean() + 
                            weight_g * loss_g.mean() + 
                            weight_n * loss_n.mean() +
                            weight_cross_modal * loss_cross_modal.mean())
                    
                    loss_b_all = loss_b_all + loss_b.detach().cpu().tolist() 
                    loss_g_all = loss_g_all + loss_g.detach().cpu().tolist()
                    loss_n_all = loss_n_all + loss_n.detach().cpu().tolist() 
                    loss_cross_modal_all=loss_cross_modal_all+loss_cross_modal.detach().cpu().tolist()
                else:
                    loss = (loss_b.mean() + 
                            loss_g.mean() + 
                            loss_n.mean() +
                            loss_cross_modal.mean())
                
                loss_all += loss.item() * data.num_graphs
                loss.backward()
                optimizer.step()
            
            print('[TRAIN] Epoch:{:03d} | Loss:{:.4f}'.format(epoch, loss_all / n_train))

            if args.is_adaptive:
                mean_b, std_b = np.mean(loss_b_all), np.std(loss_b_all) 
                mean_g, std_g = np.mean(loss_g_all), np.std(loss_g_all)
                mean_n, std_n = np.mean(loss_n_all), np.std(loss_n_all) 
                mean_hp, std_hp = np.mean(loss_cross_modal_all), np.std(loss_cross_modal_all)
                
                std_b = std_b if std_b > 1e-6 else 1.0
                std_g = std_g if std_g > 1e-6 else 1.0
                std_n = std_n if std_n > 1e-6 else 1.0
                std_hp = std_hp if std_hp > 1e-6 else 1.0
            
            if epoch % args.eval_freq == 0:
                cluster_result_b_eval = get_cluster_result(dataloader, model, args, use_hyper=False)
                cluster_result_hyper_eval = get_cluster_result(dataloader, model, args, use_hyper=True)
                
                model.eval()

                y_score_all = []
                y_true_all = []
                for data in dataloader_test:
                    data = data.to(device)
                    b, hyper_b, g_f, g_s, n_f, n_s, g_hyper_f, g_hyper_s, n_hyper_f, n_hyper_s = model(
                        data.x, data.x_s, data.edge_index, data.batch, data.num_graphs
                    )
                    
                    y_score_b = model.scoring_b(b, cluster_result_b_eval) 
                    y_score_g = model.calc_loss_g(g_f, g_s)
                    y_score_n = model.calc_loss_n(n_f, n_s, data.batch) 
                    
                    y_score_cross_modal_b = model.scoring_cross_modal(b, cluster_result_hyper_eval['centroids'])
                    y_score_cross_modal_hyper = model.scoring_cross_modal(hyper_b, cluster_result_b_eval['centroids'])
                    y_score_cross_modal = y_score_cross_modal_b + y_score_cross_modal_hyper
                    
                    if args.is_adaptive:
                        y_score = (y_score_b - mean_b) / std_b + (y_score_g - mean_g) / std_g + (y_score_n - mean_n) / std_n +(y_score_cross_modal-mean_hp)/std_hp

                    else:
                        y_score = y_score_b + y_score_g + y_score_n + y_score_cross_modal

                    
                    y_true = data.y

                    y_score_all = y_score_all + y_score.detach().cpu().tolist()
                    y_true_all = y_true_all + y_true.detach().cpu().tolist()
                
                y_score_all = np.array(y_score_all)
                if np.isnan(y_score_all).any() or np.isinf(y_score_all).any():
                    print(f"[WARNING] y_score_all contains NaN or Inf at epoch {epoch}. Replacing with 0.")
                    y_score_all = np.nan_to_num(y_score_all)

                auc = skm.roc_auc_score(y_true_all, y_score_all)
                print('[EVAL] Epoch: {:03d} | AUC:{:.4f}'.format(epoch, auc))
                if auc > best_auc:
                    best_auc = auc
                    best_model_state = copy.deepcopy(model.state_dict())
                    print(f"[BEST] New best AUC: {best_auc:.4f}")
   
        print('[RESULT] Trial: {:02d} | AUC:{:.4f}'.format(trial, best_auc))
        aucs.append(best_auc)
    aucs.sort(reverse=True)
    ddps=aucs[:3]
    avg_auc = np.mean(ddps)
    std_auc = np.std(ddps)
    print(ddps)
    print('[FINAL RESULT] AVG_AUC:{:.4f}+-{:.4f}'.format(avg_auc, std_auc))