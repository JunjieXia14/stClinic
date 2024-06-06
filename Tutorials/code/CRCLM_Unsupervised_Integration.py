import warnings
warnings.filterwarnings("ignore")

import anndata
import scanpy as sc
import random
from stClinic.mnn_utils import *
from stClinic.Utilities import *
from stClinic.Module import *

import torch
used_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(used_device)

def extract_number(s):
    first_sort = int(re.findall(r'\d+', s)[0])
    second_sort = int(re.findall(r'\d+', s)[1])
    return (first_sort, second_sort)

from pathlib import Path
path = Path("CRCLM")
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
parser.add_argument('--rad_cutoff',   '-RC', type = int, default = 100,    help='The radius value used to construct intra-edge (i.e., spatial nearest neighbors) for each slice')
parser.add_argument('--k_cutoff',   '-KC', type = int, default = 6,    help='The number of spatial neighbors used to construct intra-edge')
parser.add_argument('--k',   '-K', type = int, default = 5,    help='The number of mutual nearest neighbors used to construct inter-edges across slices')
parser.add_argument('--n_top_genes',   '-NTG', type = int, default = 18000,    help='The number of highly variable genes selected for each slice')
parser.add_argument('--n_centroids',   '-NCS', type = int, default = 8,    help='The number of components of the GMM')
parser.add_argument('--lr_integration',   '-LRI', type = float, default = 0.0005,    help='The learning rate used by stClinic when extracting batch-corrected features in slices')
args = parser.parse_known_args()[0]

# Load data
rad_list = [100, 40, 250, 50, 200]
source_ids = ['Villemin et al', 'Valdeolivas et al', 'Wu et al', 'Garbarino et al', 'Wang et al']
print(source_ids)
Batch_list = []
adj_list = []
section_ids = []

idx = 0
for source_id, rad in zip(source_ids, rad_list):

    input_dir = os.path.join(args.input_dir, source_id)

    for type in ['CRC', 'LM']:

        h5ad_list = [h5ad_file for h5ad_file in os.listdir(input_dir) if type in h5ad_file]
        h5ad_list = sorted(h5ad_list, key=extract_number)

        if len(h5ad_list) > 0:
            print(h5ad_list)

            for ad in h5ad_list:
                idx += 1

                # Read h5ad file
                subset_dir = os.path.join(input_dir, ad)
                adata = sc.read_h5ad(subset_dir)
                adata.var_names_make_unique(join='++')
                adata.obs['batch_name_idx'] = idx

                # Set sample label
                if type=='CRC':
                    adata.obs['type'] = 'Primary'
                else:
                    adata.obs['type'] = 'Metastasis'

                # Make spot name unique
                adata.obs_names = [x + '_' + source_id[:-6] + '_' + ad[:-5] for x in adata.obs_names]

                # Construct intra-edges
                Cal_Spatial_Net(adata, rad_cutoff=rad)

                # Normalization
                sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=args.n_top_genes)
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
                adata = adata[:, adata.var['highly_variable']]

                sc.tl.pca(adata, n_comps=10, random_state=seed)

                adj_list.append(adata.uns['adj'])
                Batch_list.append(adata)
                section_ids.append(idx)

# Concat scanpy objects
section_ids = ['slice' + str(x) for x in section_ids]
adata_concat = anndata.concat(Batch_list, label="slice_name", keys=section_ids)
adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
print('\nShape of concatenated AnnData object: ', adata_concat.shape)

# Construct unified graph
mnn_dict = create_dictionary_mnn(adata_concat, use_rep='X_pca', batch_name='batch_name', k=args.k)
adj_concat = inter_linked_graph(adj_list, section_ids, mnn_dict)
adata_concat.uns['adj'] = adj_concat
adata_concat.uns['edgeList'] = np.nonzero(adj_concat)

# Run stClinic for unsupervised integration
centroids_num = args.n_centroids
print(f'Estimated centroids number: {centroids_num}')
adata_concat = train_Integration_Model(adata_concat, n_centroids=centroids_num, lr=args.lr_integration/14, device=used_device)

# Clustering and UMAP reduction
sc.pp.neighbors(adata_concat, use_rep='stClinic', random_state=seed)
sc.tl.louvain(adata_concat, resolution=1, random_state=seed)
sc.tl.umap(adata_concat, random_state=seed)

# Save AnnData object    (only X, obs & obsm)
del adata_concat.uns; del adata_concat.obsp
adata_concat.write(path / f'integrated_adata_CRCLM24.h5ad', compression='gzip')  