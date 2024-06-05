import warnings
warnings.filterwarnings("ignore")

import scanpy as sc
import pandas as pd
import numpy as np
import random
from stClinic.Utilities import *
from stClinic.Module import *

import torch
used_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(used_device)

def extract_number(s):
    return int(re.findall(r'\d+', s)[0])

from pathlib import Path
path = Path("CRCLM")
path.mkdir(parents=True, exist_ok=True)

# Set seed
seed = 666
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Load data
adata = sc.read_h5ad(path / 'integrated_adata_CRCLM24.h5ad')
adata.obs['louvain'] = adata.obs['louvain'].astype('int')
adata.obs['louvain'] = adata.obs['louvain'].astype('category')

# Data preparation
sorted_batch = sorted(np.unique(adata.obs['batch_name']), key=extract_number)

# 6 statistics measures per cluster
adata = stClinic_Statistics_Measures(adata, sorted_batch)

# Clinical information (One-hot encoding)
All_type = []
for bid in sorted_batch:
    batch_obs = adata.obs[ adata.obs['batch_name'] == bid ]
    All_type.append( np.unique( batch_obs['type'] )[0] )
All_type = np.array(All_type)
type_idx = np.zeros([len(All_type)], dtype=int)
type_idx[All_type == 'Metastasis'] = 1
adata.uns['grading'] = type_idx

# Run stClinic for supervised prediction
adata = train_Prediction_Model(adata, pred_type='grading', lr=0.05, device=used_device)

# Save AnnData object
adata.write(path / f'integrated_adata_CRCLM24.h5ad', compression='gzip')  