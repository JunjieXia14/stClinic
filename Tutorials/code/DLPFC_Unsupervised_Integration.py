import warnings
warnings.filterwarnings("ignore")

import os
os.environ['R_HOME'] = '/home/zuocm/anaconda3/envs/stClinic/lib/R'
os.environ['R_USER'] = '/home/zuocm/anaconda3/envs/stClinic/lib/python3.8/site-packages/rpy2'

import anndata
import scanpy as sc
import random
from stClinic.mnn_utils import *
from stClinic.Utilities import *
from stClinic.Module import *

import torch
used_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(used_device)

from pathlib import Path
path = Path("DLPFC")
path.mkdir(parents=True, exist_ok=True)

# Set seed
seed = 666
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Set parameters
import argparse
parser = argparse.ArgumentParser(description='stClinic')
parser.add_argument('--input_dir',   '-IP', type = str, default = '/home/zuocm/Share_data/xiajunjie/stClinic/Datasets/DLPFC/',    help='data directory')
parser.add_argument('--rad_cutoff',   '-RC', type = int, default = 150,    help='The radius value used to construct intra-edge (i.e., spatial nearest neighbors) for each slice')
parser.add_argument('--k_cutoff',   '-KC', type = int, default = 6,    help='The number of spatial neighbors used to construct intra-edge')
parser.add_argument('--k',   '-K', type = int, default = 5,    help='The number of mutual nearest neighbors used to construct inter-edges across slices')
parser.add_argument('--n_top_genes',   '-NTG', type = int, default = 5000,    help='The number of highly variable genes selected for each slice')
parser.add_argument('--n_centroids',   '-NCS', type = int, default = 7,    help='The number of components of the GMM')
parser.add_argument('--lr_integration',   '-LRI', type = float, default = 0.0005,    help='The learning rate used by stClinic when extracting batch-corrected features in slices')
args = parser.parse_known_args()[0]

# Load data
section_ids = ['151673','151674','151675','151676']
print(section_ids)
Batch_list = []
adj_list = []

for idx, section_id in enumerate(section_ids):

    # Read h5 file
    input_dir = os.path.join(args.input_dir, section_id)
    adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique(join="++")

    # Read corresponding annotation file
    Ann_df = pd.read_csv(os.path.join(input_dir, section_id + '_annotation.txt'), sep='\t', header=0, index_col=0)
    Ann_df.loc[Ann_df['Layer'].isna(),'Layer'] = "unknown"
    adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Layer'].astype('category')
    adata.obs['batch_name_idx'] = idx

    # Make spot name unique
    adata.obs_names = [x+'_'+section_id for x in adata.obs_names]

    # Construct intra-edges
    Cal_Spatial_Net(adata, rad_cutoff=args.rad_cutoff)

    # Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=args.n_top_genes)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]

    sc.tl.pca(adata, n_comps=10, random_state=seed)

    adj_list.append(adata.uns['adj'])
    Batch_list.append(adata)

# Concat scanpy objects
adata_concat = anndata.concat(Batch_list, label="slice_name", keys=section_ids)
adata_concat.obs['Ground Truth'] = adata_concat.obs['Ground Truth'].astype('category')
adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
print('\nShape of concatenated AnnData object: ', adata_concat.shape)

# Construct unified graph
# mnn_dict = create_dictionary_mnn(adata_concat, use_rep='X_pca', batch_name='batch_name', k=1) # k=0
adj_concat = inter_linked_graph(adj_list, section_ids, mnn_dict=None)
adata_concat.uns['adj'] = adj_concat
adata_concat.uns['edgeList'] = np.nonzero(adj_concat)

# Run stClinic for unsupervised integration
centroids_num = args.n_centroids
print(f'Estimated centroids number: {centroids_num}')
adata_concat = train_Integration_Model(adata_concat, n_centroids=centroids_num, lr=args.lr_integration, device=used_device)

# Clustering
mclust_R(adata_concat, num_cluster=len(np.unique(adata_concat.obs[adata_concat.obs['Ground Truth']!='unknown']['Ground Truth'])), used_obsm='stClinic')
adata_concat = adata_concat[adata_concat.obs['Ground Truth']!='unknown']

# UMAP reduction
sc.pp.neighbors(adata_concat, use_rep='stClinic', random_state=seed)
sc.tl.umap(adata_concat, random_state=seed)

# Save AnnData object    (only X, obs & obsm)
del adata_concat.uns; del adata_concat.obsp

section_str = '_'.join(section_ids)
adata_concat.write(path / f'integrated_adata_{section_str}.h5ad', compression='gzip')