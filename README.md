# stClinic


*stClinic predicts clinically relevant tumor microenvironments using spatial omics data.*


![image](https://github.com/JunjieXia14/stClinic/blob/main/image/Overview.png)

## Overview

stClinic is a tool for predicting clinically relevant tumor microenvironments from spatial omics data.


**a.** Given omics profiles (*X*) and spatial location (*S*) data across multiple slices as the input, stClinic is able to learn batch-corrected features (*z*) in unsupervised manner, and predict clinically relevant TMEs under supervision of clinical information (*Y*). **b.** stClinic utilizes a VGAE (consisting of a GAT encoder and *L* one-layer slice specific decoders) to transform *X* and a unified graph (including both intra-edges for spatially nearest spots within each slice and inter-edges for omics-similar spots across different slices) into latent features (*z*) on the GMM manifold, and repeatedly removes associations between any two spots from different GMM components to eliminate impact of false positive relations between them. **c.** stClinic adopts six statistical measures in two-dimensional UMAP space to quantify the $k_{th}$ cluster, and fuses them to characterize the representations of the $i_{th}$ slice ($r_{i}$) using attention, then learns the weight ($W^T$) of different clusters on clinical outcomes from a FC layer with SoftMax or Cox layer under supervision of sample labels. **d.** The joint low-dimensional features *z* and weight ($W^T$) of different clusters on clinical outcomes can be used for visualization, data denoising, identifing slice-specific TMEs, and predicting condition-specific TME.
