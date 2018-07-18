import numba
import matplotlib
#matplotlib.use('agg') # plotting backend compatible with screen
from .. import api as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging as logg
import os
import loompy
import scipy
import re
import anndata

def rna_velocity(adata,loomfile,basis='tsne',prefiltered=True,k=100,cleanObsRegex="-[0-9]"):
    def checkDuplicates(a):
        return(len(a) == len(set(a)))

    def gsub(regex, sub, l):
        return([re.sub(regex, sub, x) for x in l])

    def orderIntersectLists(a,b):
        set(a).intersection(b)

    def addCleanObsNames(adata,regex="-[0-9]"):
        try:
            adata.obs.loc[:,"clean_obs_names"]=adata.obs.loc[:,"clean_obs_names"]
        except:
            adata.obs.loc[:,"clean_obs_names"]=adata.obs_names
        adata.obs.loc[:,"clean_obs_names"]=gsub(regex, "",adata.obs.loc[:,"clean_obs_names"])
        return(adata)

    def match(a,b):
        return([ b.index(x) if x in b else None for x in a ])

    def openVelocyto(adata,loomfile):
        print('adding velocyto')
        ds=loompy.connect(loomfile)
        row_attrs = pd.DataFrame.from_dict(dict(ds.row_attrs.items()))
        col_attrs = pd.DataFrame.from_dict(dict(ds.col_attrs.items()))

        col_attrs.loc[:,"clean_obs_names"] = cell_names = gsub("^[a-zA-Z0-9_]+:", "",col_attrs.loc[:,'CellID'])
        col_attrs.loc[:,"clean_obs_names"] = cell_names = gsub("x","",col_attrs.loc[:,'clean_obs_names'])

        cell_names=list(adata.obs.loc[:,'clean_obs_names'])
        cell_names = [cell for cell in list(col_attrs.loc[:,'clean_obs_names']) if cell in cell_names]
        cell_index = match(list(col_attrs.loc[:,'clean_obs_names']),list(adata.obs.loc[:,'clean_obs_names']))

        gene_names = [gene for gene in row_attrs.loc[:,'Accession'] if gene in adata.var.loc[:,'gene_ids']]
        gene_index = match(list(row_attrs.loc[:,'Accession']),list(adata.var.loc[:,'gene_ids']))

        #cols_to_use = col_attrs.columns.difference( adata.obs.columns)
        adata.obs = pd.merge( adata.obs,col_attrs,how='left',on="clean_obs_names")
        adata.obs.set_index('clean_obs_names',drop=False)

        #cols_to_use = row_attrs.columns.difference(adata.vars.columns)
        vardata=pd.merge( adata.var,row_attrs,how='left',right_on="Accession",left_on="gene_ids")
        adata.var=vardata
        print(adata.var)
        adata.var.set_index('Gene',drop=False)

        if len(cell_names) == 0:
            raise ValueError(
                'Cell names in loom file do not match cell names in AnnData.')

        from anndata.base import _normalize_index
        gene_index=[x for x in gene_index if x is not None]
        cell_index=[x for x in cell_index if x is not None]
        norm_gene_index=_normalize_index(gene_index, pd.RangeIndex(len(gene_index)))
        norm_cell_index=_normalize_index(cell_index, pd.RangeIndex(len(cell_index)))

        #adata.S = anndata.AnnData(ds.layer['spliced'].sparse().tocsr().T)
        #adata.U = anndata.AnnData(ds.layer['unspliced'].sparse().tocsr().T)
        # subset to cells and genes present in adata. Do this with lists of integer indices
        adataS = anndata.AnnData(ds.layer['spliced'].sparse(norm_gene_index,norm_cell_index).tocsr().T)
        adataU = anndata.AnnData(ds.layer['unspliced'].sparse(norm_gene_index,norm_cell_index).tocsr().T)
        adataS.obs=pd.merge( adata.obs,col_attrs,how='left',on="clean_obs_names")
        adataU.obs=pd.merge( adata.obs,col_attrs,how='left',on="clean_obs_names")
        adataS.obs.set_index('clean_obs_names', drop=False)
        adataU.obs.set_index('clean_obs_names', drop=False)
        adataS.var=vardata
        adataU.var=vardata
        adataS.var.set_index('Gene', drop=False)
        adataU.var.set_index('Gene', drop=False)
        adataU.var_names=adataU.var.loc[:,"Gene"]
        adataS.var_names=adataS.var.loc[:,"Gene"]

        ds.close()
        return(adata,adataS,adataU)



    def runVelocity(adata,ad_s,ad_u,prefiltered=True):
        if prefiltered:
            subset = list(range(adata.shape[1]))#pd.RangeIndex(len(adata.var_names))
        else:
            subset, _ = sc.pp.filter_genes(ad_u.X, min_cells=10)
            ad_s = ad_s[:, subset]
            ad_u = ad_u[:, subset]

        #ad_s.var_names = np.array(ad_u.var.loc[:,"Gene"])[subset]

        # loop over genes
        from scipy.sparse import dok_matrix
        offset = np.zeros(ad_s.shape[1], dtype='float32')
        gamma = np.zeros(ad_s.shape[1], dtype='float32')
        X_du = dok_matrix(ad_s.shape, dtype='float32')
        for i in range(ad_s.shape[1]):
            x = ad_s.X[:, i].toarray()
            y = ad_u.X[:, i].toarray()
            subset = np.logical_and(x > 0, y > 0)
            x = x[subset]
            y = y[subset]
            X = np.c_[np.ones(len(x)), x]
            offset[i], gamma[i] = np.linalg.pinv(X).dot(y)
            subset_indices = np.flatnonzero(subset)
            index = subset_indices, np.array([i for dummy in subset_indices])
            X_du[index] = y - gamma[i]*x - offset[i]
            #plt.scatter(x, y)
            #plt.scatter(x, gamma[i]*x + offset[i])
            #plt.scatter(x, X_du[index].toarray()[0])
            #plt.show()
        X_du = X_du.tocoo().tocsr()

        #Need to run this outside
        #sc.pp.neighbors(adata[:,subset], n_neighbors=100)
        #Also moved this outside
        #graph = compute_velocity_graph(adata, ad_u, X_du)
        return(adata,ad_u,X_du)

    def compute_velocity_graph(adata, adata_u, X_du):
        if (adata.shape[0] != adata_u.shape[0]
            or adata_u.shape[0] != X_du.shape[0]
            or X_du.shape[0] != adata.shape[0]):
            raise ValueError('Number of cells do not match.')

        from ..neighbors import Neighbors, get_indices_distances_from_sparse_matrix
        neigh = Neighbors(adata)
        knn_indices, knn_distances = get_indices_distances_from_sparse_matrix(
            neigh.distances, neigh.n_neighbors)
        n_obs = adata.n_obs
        n_neighbors = neigh.n_neighbors

        from numpy.linalg import norm
        X_u = adata_u.X.toarray()
        X_du = X_du.astype('float32').toarray()

        def fill_graph():
            rows = np.zeros((n_obs * n_neighbors), dtype=np.int64)
            cols = np.zeros((n_obs * n_neighbors), dtype=np.int64)
            vals = np.zeros((n_obs * n_neighbors), dtype=np.float32)
            for i in range(n_obs):
                if i % 1000 == 0:
                    try:
                        logg.info('{}/{},'.format(i, n_obs))
                    except:
                        print(i,n_obs)
                for nj in range(n_neighbors):
                    j = knn_indices[i, nj]
                    if j != i:
                        du_i = X_du[i]
                        du_ji = X_u[j] - X_u[i]
                        subset = np.logical_or(du_ji != 0, du_i != 0)
                        du_i = du_i[subset]
                        du_ji = du_ji[subset]
                        # dividing this by norm(du_i) doesn't make much of a difference
                        val = np.dot(du_ji, du_i) / norm(du_ji) / norm(du_i)
                        # if val > 0, this means transitioning from i to j,
                        # convention of standard stochastic matrices even though
                        # this isn't one
                        # the problem with velocities at the boundaries of the knn
                        # graph is that, no matter in which direction they point,
                        # they anticorrelate with all neighbors: hence, one will
                        # always observe "out-going" velocity even if there is no
                        # indication for that
                        if True:  # val > 0:
                            rows[i * n_neighbors + nj] = j
                            cols[i * n_neighbors + nj] = i
                            vals[i * n_neighbors + nj] = val
            return rows, cols, vals

        rows, cols, vals = fill_graph()
        from scipy.sparse import coo_matrix
        graph = coo_matrix((vals, (rows, cols)), shape=(n_obs, n_obs))
        graph.eliminate_zeros()

        return(graph.tocsr())

    #Run above functions
    adata=addCleanObsNames(adata,cleanObsRegex)
    adata,adataS,adataU=openVelocyto(adata,loomfile)
    adata,ad_u,ad_X= runVelocity(adata,adataS,adataU)
    sc.pp.neighbors(adata, n_neighbors=k)
    adata.uns['graph']=compute_velocity_graph(adata,ad_u,ad_X)
    #adata=compute_arrows_embedding(adata,basis=basis)

    return(adata)

def compute_arrows_embedding(adata,basis="tsne"):
    if 'graph' not in adata.uns:
        raise ValueError('`arrows=True` requires `tl.rna_velocity` to be run before.')
    adjacency = adata.uns['graph']
    # loop over columns of adjacency, this is where transitions start from
    from numpy.linalg import norm
    V = np.zeros((adjacency.shape[0],  adata.obsm['X_' + basis].shape[1]), dtype='float32')
    for i, n in enumerate(adjacency.T):  # loop over columns (note the transposition)
        for j in n.nonzero()[1]:  # these are row indices
            diff = adata.obsm['X_' + basis][j] - adata.obsm['X_' + basis][i]
            # need the normalized difference vector: the distance in the embedding
            # might be completely meaningless
            diff /= norm(diff)
            V[i] += adjacency[j, i] * diff
    logg.info('added \'V_{}\' to `.obsm`'.format(basis))
    adata.obsm['V_' + basis] = V
    return(adata)

#Adapted from velocyto
def plot_velocity_arrows(adata,basis='tsne',cluster='louvain',cluster_colors='louvain_colors'):
    plt.figure(None,(14,14))
    quiver_scale = 2000
    #print(adata.uns['louvain_colors'][np.array([int(x) for x in adata.obs['louvain']])])
    cluster_colors=adata.uns['louvain_colors'][np.array([int(x) for x in adata.obs['louvain']])]
    quiver_kwargs=dict(headaxislength=7, headlength=11,
                       headwidth=8,linewidths=0.25, width=0.004,edgecolors="k",
                       color=cluster_colors, alpha=1)
    plt.quiver(adata.obsm['X_' + basis][:,0],adata.obsm['X_' + basis][:,1],
               adata.obsm['V_' + basis][:,0],adata.obsm['V_' + basis][:,1],
               scale=quiver_scale,**quiver_kwargs)
