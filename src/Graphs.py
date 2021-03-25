"""Functions for creating Chromatin Structure Graphs from HiC Data"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import cooler
import networkx as nx
import numpy as np
import pandas as pd

from .utils.parse_cooler import (
    fetch_bins_from_cooler,
    get_unique_bins,
)

from .utils.misc import (
    make_chromo_onehot,
    rename_nodes
)

log = logging.getLogger(__name__)


def initialise_graph_with_metadata(cooler_file, region1, region2) -> nx.Graph:
    return nx.Graph(cooler_file=cooler_file, region1=region1, region2=region2)


def compute_nx_graph_from_regions(
    contacts: cooler,
    regions: Dict[str, np.ndarray],
    names: Optional[dict] = {},
    balance: Optional[bool] = False,
    force_disjoint: Optional[bool] = False,
    join: Optional[bool] = False,
    backbone: Optional[bool] = True
) -> nx.Graph:
    """
    Computes a HiC Graph from a cooler file
    :param contacts: cooler file generated from a Hi-C experiment.
    :param regions: Dictionary specifying chromosomes and regions to collect data over. Dictionary should contain chromosomes as keys and 2D integer numpy arrays as values.
    :params balance: Optional boolean to determine whether returned weights should be balanced or not.
    :param force_disjoint: Optional boolean to determine whether to force the input regions to be disjoint regions.
    :param join: Optional boolean to determine whether to compose the list of graphs into a single graph object by including trans (inter-region) interactions.
    :param backbone: Optional boolean to identify edges which make up the chromatin backbone and include this as an edge feature.
    :return: nx.Graph of Hi-C Contacts
    """
    
    if join and not force_disjoint:
        print("Can't join together sub-graphs when force_disjoint=False. Risks ambiguous node and edge assignment. Setting force_disjoint = True")
        force_disjoint = True

    c = cooler.Cooler(contacts)

    # Fetch relevant bin_ids from the cooler file
    b_ids, n_ids = fetch_bins_from_cooler(cooler=c, regions=regions, names=names)
    if force_disjoint:
        # Identify unique bin_ids and isolate disjoint regions
        slices = get_unique_bins(b_ids=b_ids)
        names = []
        for sl in slices:
            # Chromosome, start, end, bins and node names for region 1
            cl = c.bins()[sl[0]]["chrom"].values[0]
            stl = c.bins()[sl[0]]["start"].values[0]
            el = c.bins()[sl[-1] + 1]["end"].values[0]

            sl_id = f"{cl}:{stl}-{el}"
            n_ids.append(sl_id)
    else:
        slices = [np.array(item) for item in b_ids]

    # Initialise Graph List
    Glist = []
    for idx, s1 in enumerate(slices):
        for jdx, s2 in enumerate(slices[idx:]):
            if s1[0] != s2[0] and not join:
                continue

            # Chromosome, start, end, bins and node names for region 1
            c1 = c.bins()[s1[0]]["chrom"].values[0]
            st1 = c.bins()[s1[0]]["start"].values[0]
            e1 = c.bins()[s1[-1] + 1]["end"].values[0]

            s1_id = f"{c1}:{st1}-{e1}"
            s1_id = n_ids[idx]

            b1 = c.bins()[s1[0] : s1[-1] + 1]
            n1 = b1.apply(lambda x: f"{x[0]}:{x[1]}-{x[2]}", axis=1).values

            # Chromosome, start, end, bins and node names for region 2
            c2 = c.bins()[s2[0]]["chrom"].values[0]
            st2 = c.bins()[s2[0]]["start"].values[0]
            e2 = c.bins()[s2[-1] + 1]["end"].values[0]

            s2_id = f"{c2}:{st2}-{e2}"
            s2_id = n_ids[jdx]

            b2 = c.bins()[s2[0] : s2[-1] + 1]
            n2 = b1.apply(lambda x: f"{x[0]}:{x[1]}-{x[2]}", axis=1).values

            # Create graph and add unique bins as nodes
            G = initialise_graph_with_metadata(cooler_file=contacts, 
                                               region1=n_ids[idx], 
                                               region2=n_ids[jdx])

            if s1_id == s2_id:
                unique_bins = b1.index.values
                unique_nodes = n1
            else:
                unique_bins = np.append(b1.index.values, b2.index.values)
                unique_nodes = np.append(n1, n2)

            G.add_nodes_from(unique_bins)

            nx.set_node_attributes(
                G, dict(zip(unique_bins, unique_nodes)), "bin_regions"
            )

            
            mat = c.matrix(balance=balance, sparse=True)[
                s1[0] : s1[-1] + 1, s2[0] : s2[-1] + 1
            ]
            
            edge_data = np.concatenate(
                [
                    s1[mat.row][:, None],
                    s2[mat.col][:, None],
                    mat.data[:, None],
                ],
                axis=1,
            )
            
            G.add_weighted_edges_from(
                [(int(row[0]), int(row[1]), row[2]) for row in edge_data]
            )

            if s1_id == s2_id and backbone:
                bbone_edges = [(b_id, b_id + 1) for b_id in s1[:-1]]
                not_included = []
                
                try:
                    mean_weight = np.percentile([item[2] for item in G.edges.data('weight')],90)
                except:
                    mean_weight = 1
                
                for edge in bbone_edges:
                    if edge in G.edges:
                        G[edge[0]][edge[1]]["backbone"] = True
                    else:
                        G.add_edge(edge[0], edge[1], 
                                   weight=mean_weight,
                                   backbone=True)

            Glist.append(G)
    
    #if the trans option is specified then we essentially are looking at joining disjoint cis maps
    #so we can output the result as a single graph
    if join:
        out_G = Glist[0]
        g_ids = [out_G.graph['region1']]
        if len(Glist[0])>1:
            for item in Glist[1:]:
                r1 = item.graph['region1']
                r2 = item.graph['region2']
                if r1 not in g_ids:
                    g_ids.append(r1)
                if r2 not in g_ids:
                    g_ids.append(r2)
                    
                out_G = nx.compose(out_G, item) 
        out_G.graph = {'cooler_file': contacts,
                       'regions': g_ids}
        return out_G
    else:
        return Glist
    
#compute pytorch geometric data object from regions
def compute_ptg_graph_from_regions(
    contacts: cooler,
    regions: Dict[str, np.ndarray],
    names: Optional[dict] = {},
    balance: Optional[bool] = False,
    join: Optional[bool] = False,
    force_disjoint: Optional[bool] = False,
    backbone: Optional[bool] = True,
    record_cistrans_interactions: Optional[bool] = True,
    record_backbone_interactions: Optional[bool] = True,
    record_node_chromosome_as_onehot: Optional[bool] = True,
    chromosomes: Optional[list] = ["chr{}".format(str(i+1)) for i in np.arange(19)] + ['chrX']
) -> list:
    """
    Computes a HiC Graph from a cooler file
    :param contacts: cooler file generated from a Hi-C experiment.
    :param regions: Dictionary specifying chromosomes and regions to collect data over. Dictionary should contain chromosomes as keys and 2D integer numpy arrays as values.
    :param names: Dictionary specifying the name associated with each region
    :params balance: Optional boolean to determine whether returned weights should be balanced or not.
    :param join: Optional boolean to determine whether trans (inter-region) interactions are included. If this option is True then we automatically compose the subgraphs into one big graph
    :param force_disjoint: Optional boolean to determine whether to force the input regions to be disjoint regions.
    :param backbone: Optional boolean to identify edges which make up the chromatin backbone and include this as an edge feature.
    :param record_cistrans_interactions: Optional boolean to explicitely record cis (within chromosome) and trans (between chromosome) interactions within the edge_attributes
    :param record_backbone_interactions: Optional boolean to explicitely record backbone interactions within the edge attributes
    :param record_node_chromosome_as_onehot: Optional boolean to explicitely record node chromosome as a feature vector with chromosomes one hot encoded.
    :param chromosomes: Optional list of chromosomes with which to perform the one-hot encoding
    :return: edge index object, edge attributes and node assignments by chromosome 
    """
    if join and not force_disjoint:
        print("Can't join together sub-graphs when force_disjoint=False. Risks ambiguous node and edge assignment. Setting force_disjoint = True")
        force_disjoint = True
        
    c = cooler.Cooler(contacts)

    # Fetch relevant bin_ids from the cooler file
    b_ids, n_ids = fetch_bins_from_cooler(cooler=c, regions=regions, names=names)
    if force_disjoint:
        # Identify unique bin_ids and isolate disjoint regions
        slices = get_unique_bins(b_ids=b_ids)
        n_ids = []
        for sl in slices:
            # Chromosome, start, end, bins and node names for region 1
            cl = c.bins()[sl[0]]["chrom"].values[0]
            stl = c.bins()[sl[0]]["start"].values[0]
            el = c.bins()[sl[-1] + 1]["end"].values[0]

            sl_id = f"{cl}:{stl}-{el}"
            n_ids.append(sl_id)
    else:
        slices = [np.array(item) for item in b_ids]
    
    
    # Iterate through slices, adding in edge indexes and edge attributes
    edge_idxs = []
    edge_attrs = []
    sub_graph_nodes = []
    for idx, s1 in enumerate(slices):
        for s2 in slices[idx:]:
            if s1[0] != s2[0] and not join:
                continue
                
            mat = c.matrix(balance=balance, sparse=True)[
                s1[0] : s1[-1] + 1, s2[0] : s2[-1] + 1
            ]

            edge_index = np.concatenate(
                [
                    s1[mat.row][None,:],
                    s2[mat.col][None,:]
                ],
                axis=0,
            )
            
            edge_data = mat.data[:, None]
            
            if s1[0] == s2[0] and backbone:
                #Initialise backbone edge indices that need to be added in to complete the backbone
                bbone_index_to_add = []
                #Initialise a list to hold the weights for these added edges
                bbone_data_to_add = []
                #Identify backbone edges as those where the edge indices differ by exactly 1
                bbone_idxs = abs(np.diff(edge_index,
                                         axis = 0)
                                ) == 1
                #retrieve the edges
                bbone_edges = edge_index[
                    :,bbone_idxs[0,:]
                ]
                
                #attempt to work out the mean strength of a backbone edge
                #if not possible, initialise to 1
                try:
                    mean_weight = np.median(edge_data[bbone_idxs[0,:]])
                except:
                    mean_weight = 1
                
                #if we're explicitely recording backbone interactions
                #initalise and populate a vector to store these
                if record_backbone_interactions:
                    bbone_data = np.zeros(edge_data.shape
                                         )
                    bbone_data[bbone_idxs.T] = 1
                
                #for each possible backbone edge, check if it already exists
                #if not then add it in
                for sid in s1[:-1]:
                    bbone_edge_exists = np.sum((bbone_edges[0,:]==sid)&(bbone_edges[1,:]==sid+1))
                    if bbone_edge_exists:
                        continue
                    
                    bbone_edge_exists = np.sum((bbone_edges[0,:]==sid+1)&(bbone_edges[1,:]==sid))
                    if bbone_edge_exists:
                        print('damn daniel')
                        continue
                    
                    bbone_index_to_add.append([sid,
                                               sid+1]
                                             )
                    bbone_data_to_add.append(mean_weight)
                    
                if len(bbone_index_to_add) >0:
                    #update the edge index and edge weight arrays
                    edge_index = np.concatenate([edge_index, 
                                            np.array(bbone_index_to_add).T], 
                                            axis = 1
                                           )
                    edge_data = np.append(edge_data, 
                                          np.array(bbone_data_to_add)[:,None],
                                          axis = 0
                                     )
                    #if we're explicitely recording backbone interactions then
                    #append the backbone data to the current edge data
                    if record_backbone_interactions:
                        bbone_data = np.append(bbone_data,
                                               np.ones(len(bbone_data_to_add))[:,None],
                                               axis = 0
                                          )
            
                edge_data = np.concatenate([edge_data,
                                            bbone_data],
                                           axis = 1)
            elif record_backbone_interactions:
                bbone_data = np.zeros(edge_data.shape)
                edge_data = np.concatenate([edge_data,
                                            bbone_data],
                                           axis = 1)
            
            #if we're explicitely recording whether an interaction is cis or trans then
            #add that information into the 
            if record_cistrans_interactions:
                c1 = c.bins()[s1[0]]["chrom"].values[0]
                c2 = c.bins()[s2[0]]["chrom"].values[0]
                if c1 == c2:
                    #cis interactions encoded as a vector of zeros i.e. same chromo
                    edge_data = np.concatenate([edge_data, 
                                               np.zeros(edge_data.shape[0])[:,None]],
                                               axis = 1
                                              )
                else:
                    #cis interactions encoded as a vector of one i.e. different chromo
                    edge_data = np.concatenate([edge_data, 
                                               np.ones(edge_data.shape[0])[:,None]],
                                               axis = 1
                                              )
    
            #make edges bidirectional
            edge_index = np.append(edge_index,
                                   edge_index[::-1,:],
                                   axis = 1
                                  )
            edge_data = np.append(edge_data,
                                  edge_data,
                                  axis = 0
                                 )
            
            edge_idxs.append(edge_index)
            edge_attrs.append(edge_data)
            if s1[0] == s2[0]:
                sub_graph_nodes.append(s1)
            else:
                sub_graph_nodes.append([s1,s2])
    
    if join:
        mynodes = np.concatenate(slices)
        edge_index = np.concatenate(edge_idxs, axis = 1)
        edge_data = np.concatenate(edge_attrs, axis = 0)
        
        onehots = []
        if record_node_chromosome_as_onehot:
            for sl in slices:
                chrom = c.bins()[sl[0]]["chrom"].values[0]
                onehot_x = make_chromo_onehot(chrom, 
                                              sl.shape[0],
                                              chromosomes = chromosomes
                                             )
                onehots.append(onehot_x)
                
            onehots = np.concatenate(onehots,
                                     axis = 0
                                    )
        
        edge_index = rename_nodes(edge_index, mynodes)
        bin_info = pd.concat([c.bins()[sl[0]:sl[-1]+1] for sl in slices], axis = 0)
        out_dict = {}
        out_dict['edge_index'] = edge_index
        out_dict['edge_attrs'] = edge_data
        out_dict['x'] = onehots
        out_dict['cooler_idxs'] = bin_info.index.values 
        out_dict['name'] = n_ids
        return out_dict
    else:
        out = []
        onehots = []
        if record_node_chromosome_as_onehot:
            for sl in slices:
                chrom = c.bins()[sl[0]]["chrom"].values[0]
                onehot_x = make_chromo_onehot(chrom, 
                                              sl.shape[0],
                                              chromosomes = chromosomes
                                             )
                onehots.append(onehot_x)
        
        for idx, edge_index in enumerate(edge_idxs):
            mynodes = sub_graph_nodes[idx]
            bin_info = c.bins()[sub_graph_nodes[idx][0]:sub_graph_nodes[idx][-1]+1]
            edge_index = rename_nodes(edge_index, mynodes)
            out_dict = {}
            out_dict['edge_index'] = edge_index
            out_dict['edge_attrs'] = edge_attrs[idx]
            out_dict['x'] = onehots[idx]
            out_dict['cooler_idxs'] = bin_info.index.values
            out_dict['name'] = n_ids[idx]
            out.append(out_dict)
        
        return out


if __name__ == "__main__":
    regions = {
        "chr1": np.array([[1, 10000], [11000, 20000]]),
        "chr2": np.array([[1, 10000], [11000, 20000]]),
        "chr3": np.array([[1, 10000], [11000, 20000]]),
    }

    compute_nx_graph_from_regions(
        "Dixon2012-H1hESC-HindIII-allreps-filtered.1000kb.cool",
        regions=regions,
    )
