"""Functions for creating Chromatin Structure Graphs from HiC Data"""
import logging
from typing import Dict, Optional, List

import cooler
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix as coo

from .utils.parse_cooler import (
    make_slices,
    make_bins
)

from .utils.misc import (
    make_chromo_onehot,
    rename_nodes
)

log = logging.getLogger(__name__)


def initialise_graph_with_metadata(cooler_file, region1, region2) -> nx.Graph:
    return nx.Graph(cooler_file=cooler_file, region1=region1, region2=region2)

def compute_nx_graph_from_regions(
    contacts: cooler.Cooler,
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

    slices, n_ids = make_slices(clr=c, 
                                regions=regions, 
                                names=names, 
                                force_disjoint = force_disjoint)

    # Initialise Graph List
    Gdict = {}
    chroms = list(slices.keys())
    for cidx1, chrom1 in enumerate(chroms):
        Gdict[chrom1] = {}
        for chrom2 in chroms[cidx1:]:
            if chrom1 != chrom2 and not join:
                continue 
            Gdict[chrom1][chrom2] = []
            for idx, s1 in enumerate(slices[chrom1]):
                if chrom1 == chrom2:
                    #don't want to repeat region pairings
                    slist = slices[chrom1][idx:]
                else:
                    slist = slices[chrom2]
                
                for jdx,s2 in enumerate(slist):
                    if s1[0] != s2[0] and not join:
                        continue

                    # Chromosome, start, end, bins and node names for region 1
                    c1 = c.bins()[s1[0]]["chrom"].values[0]
                    st1 = c.bins()[s1[0]]["start"].values[0]
                    e1 = c.bins()[s1[-1] + 1]["end"].values[0]

                    s1_id = f"{c1}:{st1}-{e1}"
                    s1_id = n_ids[chrom1][idx]

                    b1 = c.bins()[s1[0] : s1[-1] + 1]
                    n1 = b1.apply(lambda x: f"{x[0]}:{x[1]}-{x[2]}", axis=1).values

                    # Chromosome, start, end, bins and node names for region 2
                    c2 = c.bins()[s2[0]]["chrom"].values[0]
                    st2 = c.bins()[s2[0]]["start"].values[0]
                    e2 = c.bins()[s2[-1] + 1]["end"].values[0]

                    s2_id = f"{c2}:{st2}-{e2}"
                    s2_id = n_ids[chrom2][jdx]

                    b2 = c.bins()[s2[0] : s2[-1] + 1]
                    n2 = b1.apply(lambda x: f"{x[0]}:{x[1]}-{x[2]}", axis=1).values

                    # Create graph and add unique bins as nodes
                    G = initialise_graph_with_metadata(cooler_file=contacts, 
                                               region1=n_ids[chrom1][idx], 
                                               region2=n_ids[chrom2][jdx])

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

                    Gdict[chrom1][chrom2].append(G)
    
    #if the trans option is specified then we essentially are looking at joining disjoint cis maps
    #so we can output the result as a single graph
    if join:
        Glist = []
        for chrom1 in Gdict:
            for chrom2 in Gdict[chrom1]:
                for graph in Gdict[chrom1][chrom2]:
                    Glist.append(graph)
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
                       'regions': n_ids}
        return out_G
    else:
        return Gdict
    
    
def make_edges_bidirectional(
    edge_index: np.ndarray,
    edge_data: np.ndarray
):
    edge_index = np.append(edge_index,
                           edge_index[::-1,:],
                           axis = 1
                          )
    
    edge_data = np.append(edge_data,
                          edge_data,
                          axis = 0
                         )
    
    return edge_index, edge_data


def add_cistrans_interactions(
    edge_data: np.ndarray,
    c1: str,
    c2: str
)->np.ndarray:
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
        
    return edge_data

def add_vector_edge_weighted_self_loops(
    edge_index: np.ndarray, 
    edge_data: np.ndarray,
    nodes: np.ndarray,
    fill_value: Optional[int] = 1
):
    idxs = np.diff(edge_index, axis = 0) == 0
    idxs = idxs[0,:]
    
    if np.sum(idxs)>0:
        vec_add = np.mean(edge_data[idxs,:],
                          axis = 0)
        already_looped = edge_index[0,idxs].astype('int32')-int(np.min(nodes))
    else:
        vec_add = np.full(edge_data.shape[1],
                          fill_value)
        already_looped = np.array([],'int32')
        
    
    notlooped = np.ones(nodes.shape[0])
    notlooped[already_looped] = 0
    
    notlooped = np.where(notlooped)[0]
    
    if notlooped.shape[0]>0:
        ei_add = np.stack([notlooped,notlooped]) + int(np.min(nodes))
        ea_add = np.stack([vec_add for i in np.arange(notlooped.shape[0])])
        
        edge_index = np.append(edge_index, ei_add, axis = 1)
        edge_data = np.append(edge_data, ea_add, axis = 0)
    
    return edge_index.astype('int'), edge_data


def add_backbone_interactions(
    edge_index: np.ndarray, 
    edge_data: np.ndarray,
    nodes: np.ndarray,
    record_backbone_interactions: Optional[bool] = True,
    add_self_loops: Optional[bool] = True
):
    #add in self loops that aren't present
    if add_self_loops:
        edge_index, edge_data = add_vector_edge_weighted_self_loops(edge_index, 
                                                                    edge_data,
                                                                    nodes
                                                                   )
    
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
        mean_weight = np.median(edge_data[bbone_idxs[0,:],:])
    except:
        mean_weight = 1
    
    
    #if we're explicitely recording backbone interactions
    #initalise and populate a vector to store these
    if record_backbone_interactions:
        bbone_data = np.zeros(edge_data.shape[0]
                             )
        bbone_data[bbone_idxs.T[:,0]] = 1
        
        self_idxs = np.diff(edge_index,
                            axis = 0) == 0
        bbone_data[self_idxs[0,:]] = 1
                
    #for each possible backbone edge, check if it already exists
    #if not then add it in
    for sid in nodes[:-1]:
        bbone_edge_exists = np.sum((bbone_edges[0,:]==sid)&(bbone_edges[1,:]==sid+1))
        if bbone_edge_exists:
            continue
                    
        bbone_edge_exists = np.sum((bbone_edges[0,:]==sid+1)&(bbone_edges[1,:]==sid))
        if bbone_edge_exists:
            continue
                    
        bbone_index_to_add.append([sid,
                                   sid+1]
                                 )
        bbone_data_to_add.append([mean_weight]*edge_data.shape[1])
        
        bbone_index_to_add.append([sid+1,
                                   sid]
                                 )
        bbone_data_to_add.append([mean_weight]*edge_data.shape[1])
                    
    if len(bbone_index_to_add) >0:
        #update the edge index and edge weight arrays
        edge_index = np.concatenate([edge_index, 
                                     np.array(bbone_index_to_add).T], 
                                    axis = 1
                                   )
        edge_data = np.append(edge_data,
                              np.array(bbone_data_to_add),
                              axis = 0
                             )
        #if we're explicitely recording backbone interactions then
        #append the backbone data to the current edge data
        if record_backbone_interactions:
            bbone_data = np.append(bbone_data,
                                   np.ones(len(bbone_data_to_add)),
                                   axis = 0
                                  )
    
    edge_data = np.concatenate([edge_data,
                                bbone_data[:,None]],
                               axis = 1)
    
    
    return edge_index, edge_data    
    

def _single_clr_edge_and_node_info_from_slices(
    c: cooler.Cooler, 
    slices: Dict[str,List[np.ndarray]], 
    balance: Optional[bool] = True, 
    join: Optional[bool] = False
):
    # Iterate through slices, adding in edge indexes and edge attributes
    edge_idxs = {}
    edge_attrs = {}
    sub_graph_nodes = {}
    chroms = list(slices.keys())
    for cidx1,chrom1 in enumerate(chroms):
        edge_idxs[chrom1] = {}
        edge_attrs[chrom1] = {}
        sub_graph_nodes[chrom1] = {}
        for chrom2 in chroms[cidx1:]:
            if chrom1 != chrom2 and not join:
                continue
            edge_idxs[chrom1][chrom2] = []
            edge_attrs[chrom1][chrom2] = []
            sub_graph_nodes[chrom1][chrom2] = []      
            for idx, s1 in enumerate(slices[chrom1]):
                if chrom1 == chrom2:
                    #don't want to repeat region pairings
                    slist = slices[chrom1][idx:]
                else:
                    slist = slices[chrom2]
                for jdx,s2 in enumerate(slist):
                    if s1[0] == s2[0] and jdx != 0:
                        continue
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
                
                    if np.sum(s1-s2)!=0:
                        edge_index, edge_data = make_edges_bidirectional(edge_index,edge_data)
                    
                    ind = np.lexsort((edge_index[0,:],edge_index[1,:]))
                    edge_index = edge_index[:,ind]
                    edge_data = edge_data[ind,:]
            
                    edge_idxs[chrom1][chrom2].append(edge_index)
                    edge_attrs[chrom1][chrom2].append(edge_data)
                    if s1[0] == s2[0]:
                        sub_graph_nodes[chrom1][chrom2].append(s1)
                    else:
                        sub_graph_nodes[chrom1][chrom2].append(np.append(s1,s2))
                        
                        
    return edge_idxs, edge_attrs, sub_graph_nodes

def _single_clr_edge_and_node_info_from_sites(
    c: cooler.Cooler, 
    sites: Dict[str,np.ndarray], 
    balance: Optional[bool] = True, 
    join: Optional[bool] = False
):
    # Iterate through slices, adding in edge indexes and edge attributes
    edge_idxs = {}
    edge_attrs = {}
    sub_graph_nodes = {}
    chroms = list(sites.keys())
    for idx, chrom1 in enumerate(chroms):
        edge_idxs[chrom1] = {}
        edge_attrs[chrom1] = {}
        sub_graph_nodes[chrom1] = {}
        for chrom2 in chroms[idx:]:
            if chrom1 != chrom2 and not join:
                continue
            mat = c.matrix(balance=balance).fetch(chrom1,
                                                  chrom2)
            mat = mat[sites[chrom1],:]
            mat = mat[:,sites[chrom2]]
                    
            mat = coo(mat)
            
            b1 = c.bins().fetch(chrom1).index.values[sites[chrom1]]
            b2 = c.bins().fetch(chrom2).index.values[sites[chrom2]]
            
            edge_index = np.concatenate(
                        [
                            b1[mat.row][None,:],
                            b2[mat.col][None,:]
                        ],
                        axis=0,
                    )
            
            edge_data = mat.data[:, None]
            
            if chrom1 != chrom2:
                edge_index = np.append(edge_index,
                                       edge_index[::-1,:],
                                       axis = 1
                                      )
                edge_data = np.append(edge_data,
                                      edge_data,
                                      axis = 0
                                     )
                
                edge_data[np.isnan(edge_data)] = 0
                
            ind = np.lexsort((edge_index[0,:],edge_index[1,:]))
            edge_index = edge_index[:,ind]
            edge_data = edge_data[ind,:]
            
            edge_idxs[chrom1][chrom2]= [edge_index]
            edge_attrs[chrom1][chrom2]=[edge_data]
        
            if chrom1 == chrom2:
                sub_graph_nodes[chrom1][chrom2] = [b1]
            else:
                sub_graph_nodes[chrom1][chrom2] = [np.append(b1,
                                                             b2)]
                        
    
    return edge_idxs, edge_attrs, sub_graph_nodes


def join_edge_attrs(
    edge_idxs: List[np.ndarray],
    edge_attrs: List[np.ndarray],
    graph_nodes: List[np.ndarray]
)->List[np.ndarray]:
    '''
    Given some list of edge index arrays and some 1D attributes for those edges, creates a new edge index array
    where the data for each edge is a vector containing the relevant data from the input list of edge attributes
    '''
    if len(edge_idxs)==1:
        return [edge_idxs[0], edge_attrs[0]]
    
    contiguous = np.sum(np.diff(graph_nodes[0])-1)==0
    if contiguous:
        for idx, nodes in enumerate(graph_nodes):
            edge_idxs[idx] -= nodes[0]
    else:
        for idx,nodes in enumerate(graph_nodes):
            edge_idxs[idx] = rename_nodes(edge_idxs[idx],
                                          nodes)
    
    numnodes = len(graph_nodes[0])
    dummy_idxs = [(numnodes*item[0,:])+item[1,:] for item in edge_idxs]
    
    
    hit_idxs = np.array(list(set().union(*dummy_idxs)))
    
    all_edges = np.vstack(
        [((hit_idxs.astype('float'))/numnodes).astype('int'),
         hit_idxs.astype('int')- numnodes*((hit_idxs.astype('float'))/numnodes).astype('int')
        ])
    
    all_edges = rename_nodes(all_edges,
                             np.arange(graph_nodes[0].shape[0]),
                             new_nodes = graph_nodes[0])
    
    num_graphs = len(edge_idxs)
    all_data = np.zeros((all_edges.shape[1],num_graphs))
    
    edge_dict = {edge: idx for idx, edge in enumerate(hit_idxs)}
    f = lambda x: edge_dict[x]
    for idx, attr in enumerate(edge_attrs):
        attr_idxs = np.array(list(map(f, dummy_idxs[idx])))
        all_data[attr_idxs.astype('int'),idx] = attr[:,0]
        
    return all_edges, all_data    

def join_multi_clr_graphs(
    clr_edge_idxs: List[np.ndarray],
    clr_edge_attrs: List[np.ndarray],
    clr_sub_graph_nodes: List[np.ndarray],
    backbone: Optional[bool] = True,
    record_backbone_interactions: Optional[bool] = True,
    record_cistrans_interactions: Optional[bool] = False,
    add_self_loops: Optional[bool] = True
):
    edge_idxs = {}
    edge_attrs = {}
    for chrom1 in clr_edge_idxs[0]:
        edge_idxs[chrom1] = {}
        edge_attrs[chrom1] = {}
        for chrom2 in clr_edge_idxs[0][chrom1]:
            edge_idxs[chrom1][chrom2] = []
            edge_attrs[chrom1][chrom2] = []
            for idx, reg_graph in enumerate(clr_edge_idxs[0][chrom1][chrom2]):
                sep_edge_idxs = [item[chrom1][chrom2][idx] for item in clr_edge_idxs]
                
                sep_edge_attrs = [item[chrom1][chrom2][idx] for item in clr_edge_attrs]
                
                sep_sub_graph_nodes = [item[chrom1][chrom2][idx] for item in clr_sub_graph_nodes]

                ei, ea = join_edge_attrs(sep_edge_idxs,
                                         sep_edge_attrs,
                                         sep_sub_graph_nodes
                                        )
                
                contiguous = np.sum(np.diff(sep_sub_graph_nodes[0])-1)==0
                if backbone and contiguous and chrom1==chrom2:
                    ei, ea = add_backbone_interactions(ei,
                                                       ea,
                                                       sep_sub_graph_nodes[0],
                                                       record_backbone_interactions = record_backbone_interactions,
                                                       add_self_loops = add_self_loops
                                                      )
                    
                    
                elif backbone and record_backbone_interactions:
                    ea = np.append(ea,
                                   np.zeros(ea.shape[0])[:,None],
                                   axis = 1
                                  )
                
                if record_cistrans_interactions:
                    ea = add_cistrans_interactions(ea,
                                                   chrom1,
                                                   chrom2)
                
                edge_idxs[chrom1][chrom2].append(ei) 
                edge_attrs[chrom1][chrom2].append(ea)
                
    return edge_idxs, edge_attrs

#compute pytorch geometric data object from regions
def compute_ptg_graph_from_regions(
    contacts: List[str],
    regions: Dict[str, np.ndarray],
    names: Optional[dict] = {},
    balance: Optional[bool] = False,
    join: Optional[bool] = False,
    force_disjoint: Optional[bool] = False,
    backbone: Optional[bool] = True,
    record_cistrans_interactions: Optional[bool] = False,
    record_backbone_interactions: Optional[bool] = True,
    record_node_chromosome_as_onehot: Optional[bool] = False,
    add_self_loops: Optional[bool] = True,
    record_names = True,
    same_index = True,
    chromosomes: Optional[list] = ["chr{}".format(str(i+1)) for i in np.arange(19)] + ['chrX']
) -> list:
    """
    Computes a HiC Graph from a list of cooler files
    :param contacts: list of cooler files generated from Hi-C experiments. Cooler files don't have to be indexed the same but they do have to all contain the chromosomes and regions being probed.
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
    
    if isinstance(contacts,str):
        contacts = [contacts]

    clr_edge_idxs = []
    clr_edge_attrs = []
    clr_sub_graph_nodes = []
    slices = []
    n_ids = []
    
    for idx,clr in enumerate(contacts):
        c = cooler.Cooler(clr)
        if idx == 0:
            sl, n_ids = make_slices(clr=c, 
                                regions=regions, 
                                names=names, 
                                force_disjoint = force_disjoint)
        elif not same_index:
            sl, n_ids = make_slices(clr=c, 
                                regions=regions, 
                                names=names, 
                                force_disjoint = force_disjoint)
        if same_index:
            slices = [sl]
        else:
            slices.append(sl)
        ei, ea, sgn = _single_clr_edge_and_node_info_from_slices(c, 
                                                                 sl,
                                                                 balance = balance,
                                                                 join = join
                                                                )
        clr_edge_idxs.append(ei)
        clr_edge_attrs.append(ea)
        clr_sub_graph_nodes.append(sgn)
    
    edge_idxs, edge_attrs = join_multi_clr_graphs(clr_edge_idxs, 
                                                  clr_edge_attrs,
                                                  clr_sub_graph_nodes,
                                                  backbone = backbone,
                                                  record_backbone_interactions = record_backbone_interactions,
                                                  record_cistrans_interactions = record_cistrans_interactions,
                                                  add_self_loops = add_self_loops
                                                 )
                
        
    if join:
        mynodes = np.concatenate([np.concatenate(sl[chrom]) for chrom in sl])
        edge_idxs = {chrom1: {chrom2: np.concatenate(edge_idxs[chrom1][chrom2],
                                                      axis = 1) for chrom2 in edge_idxs[chrom1]}for chrom1 in edge_idxs}
        edge_index = {chrom1: np.concatenate([edge_idxs[chrom1][chrom2] for chrom2 in edge_idxs[chrom1]],
                                             axis = 1) for chrom1 in edge_idxs}
        edge_index = np.concatenate([edge_index[chrom1] for chrom1 in edge_index],
                                    axis = 1)
        
        edge_attrs = {chrom1: {chrom2: np.concatenate(edge_attrs[chrom1][chrom2],
                                                      axis = 0) for chrom2 in edge_attrs[chrom1]}for chrom1 in edge_attrs}
        
        edge_data = {chrom1: np.concatenate([edge_attrs[chrom1][chrom2] for chrom2 in edge_attrs[chrom1]], 
                                             axis = 0) for chrom1 in edge_attrs}
        edge_data = np.concatenate([edge_data[chrom1] for chrom1 in edge_data],
                                    axis = 0)
        
        onehots = []
        if record_node_chromosome_as_onehot:
            for sl in mynodes[0]:
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
        bin_info = [{chrom: pd.concat([c.bins()[sl[0]:sl[-1]+1] for sl in mslice[chrom]], 
                                     axis = 0) for chrom in mslice} for mslice in slices]
        bin_info = np.stack([pd.concat([item[chrom] for chrom in item],
                             axis = 0).index.values for item in bin_info])         
        out_dict = {}
        out_dict['edge_index'] = edge_index
        out_dict['edge_attrs'] = edge_data
        out_dict['x'] = onehots
        out_dict['cooler_idxs'] = bin_info
        if record_names:
            out_dict['name'] = n_ids
        return out_dict
    else:
        c = cooler.Cooler(contacts[0])
        out = {}
        onehots = {}
        if record_node_chromosome_as_onehot:
            for chrom in slices[0]:
                onehots[chrom] = []
                for sl in slices[0][chrom]:
                    onehot_x = make_chromo_onehot(chrom, 
                                              sl.shape[0],
                                              chromosomes = chromosomes
                                             )
                    onehots[chrom].append(onehot_x)
        
        for chrom in edge_idxs:
            out[chrom] = []
            for idx, edge_index in enumerate(edge_idxs[chrom][chrom]):
                mynodes = [item[chrom][chrom][idx] for item in clr_sub_graph_nodes]
                bin_info = c.bins()[clr_sub_graph_nodes[0][chrom][chrom][idx][0]:clr_sub_graph_nodes[0][chrom][chrom][idx][-1]+1].index.values
                #print(bin_info, clr_sub_graph_nodes[0][chrom][chrom][idx])
                edge_index = rename_nodes(edge_index, mynodes[0])

                out_dict = {}
                out_dict['edge_index'] = edge_index
                out_dict['edge_attrs'] = edge_attrs[chrom][chrom][idx]
                if record_node_chromosome_as_onehot:
                    out_dict['x'] = onehots[chrom][idx]
                else:
                    out_dict['x'] = []
                out_dict['cooler_idxs'] = bin_info
                if record_names:
                    out_dict['name'] = n_ids[chrom][idx]
                out[chrom].append(out_dict)
        
        return out
    
#compute pytorch geometric data object from regions
def compute_ptg_graph_from_sites(
    contacts: List[str],
    sites: Dict[str, np.ndarray],
    names: Optional[dict] = {},
    balance: Optional[bool] = False,
    join: Optional[bool] = True,
    record_cistrans_interactions: Optional[bool] = False,
    record_node_chromosome_as_onehot: Optional[bool] = False,
    record_names: Optional[bool] = True,
    same_index: Optional[bool] = True,
    chromosomes: Optional[list] = ["chr{}".format(str(i+1)) for i in np.arange(19)] + ['chrX']
) -> list:
    """
    Computes a HiC Graph from a list of cooler files
    :param contacts: list of cooler files generated from Hi-C experiments. Cooler files don't have to be indexed the same but they do have to all contain the chromosomes and regions being probed.
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
    if isinstance(contacts,str):
        contacts = [contacts]

    clr_edge_idxs = []
    clr_edge_attrs = []
    clr_sub_graph_nodes = []
    csites = []
    
    
    for idx,clr in enumerate(contacts):
        c = cooler.Cooler(clr)
        if idx == 0:
            chrom_ind,names,bad_sites = make_bins(clr=c,
                           sites=sites,
                           names = names
                          )
        elif not same_index:
            chrom_ind,names,bad_sites = make_bins(clr=c,
                           sites=sites,
                           names = names
                          )
        if same_index:
            csites = [chrom_ind]
        else:
            csites.append(chrom_ind)
        
        ei, ea, sgn = _single_clr_edge_and_node_info_from_sites(c,
                                                                chrom_ind,
                                                                balance = balance
                                                               )
        clr_edge_idxs.append(ei)
        clr_edge_attrs.append(ea)
        clr_sub_graph_nodes.append(sgn)
    
    edge_idxs, edge_attrs = join_multi_clr_graphs(clr_edge_idxs, 
                                                  clr_edge_attrs,
                                                  clr_sub_graph_nodes,
                                                  backbone = False,
                                                  record_backbone_interactions = False,
                                                  record_cistrans_interactions = record_cistrans_interactions
                                                 )
                
        
    if join:
        mynodes = np.concatenate([np.concatenate(clr_sub_graph_nodes[0][chrom][chrom]) for chrom in sites])
        mynames = np.concatenate([names[chrom] for chrom in sites])
        
        edge_idxs = {chrom1: {chrom2: np.concatenate(edge_idxs[chrom1][chrom2],
                                                      axis = 1) for chrom2 in edge_idxs[chrom1]}for chrom1 in edge_idxs}
        edge_index = {chrom1: np.concatenate([edge_idxs[chrom1][chrom2] for chrom2 in edge_idxs[chrom1]],
                                             axis = 1) for chrom1 in edge_idxs}
        edge_index = np.concatenate([edge_index[chrom1] for chrom1 in edge_index],
                                    axis = 1)
        
        edge_attrs = {chrom1: {chrom2: np.concatenate(edge_attrs[chrom1][chrom2],
                                                      axis = 0) for chrom2 in edge_attrs[chrom1]}for chrom1 in edge_attrs}
        
        edge_data = {chrom1: np.concatenate([edge_attrs[chrom1][chrom2] for chrom2 in edge_attrs[chrom1]], 
                                             axis = 0) for chrom1 in edge_attrs}
        edge_data = np.concatenate([edge_data[chrom1] for chrom1 in edge_data],
                                    axis = 0)
        
        onehots = []
        if record_node_chromosome_as_onehot:
            for sl in mynodes[0]:
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
        out_dict = {}
        out_dict['edge_index'] = edge_index
        out_dict['edge_attrs'] = edge_data
        out_dict['x'] = onehots
        out_dict['cooler_idxs'] = mynodes
        if record_names:
            out_dict['name'] = mynames
        return out_dict, bad_sites
    else:
        c = cooler.Cooler(contacts[0])
        out = {}
        onehots = {}
        if record_node_chromosome_as_onehot:
            for chrom in sites:
                onehots[chrom] = []
                onehot_x = make_chromo_onehot(chrom, 
                                              sites[chrom].shape[0],
                                              chromosomes = chromosomes)
                onehots[chrom].append(onehot_x)
        
        for chrom in edge_idxs:
            mynodes = clr_sub_graph_nodes[0][chrom][chrom]
            edge_index = rename_nodes(edge_idxs[chrom][chrom][0], mynodes[0])
            out_dict = {}
            out_dict['edge_index'] = edge_index
            out_dict['edge_attrs'] = edge_attrs[chrom][chrom][0]
            if record_node_chromosome_as_onehot:
                out_dict['x'] = onehots[chrom][0]
            else:
                out_dict['x'] = []
            out_dict['cooler_idxs'] = mynodes
            if record_names:
                out_dict['name'] = names[chrom]
            out[chrom] = out_dict
        
        return out, bad_sites
    
    
def add_binned_data_to_graph(
    graph: dict,
    binned_data: str
) -> None:
    idxs = graph['cooler_idxs']
    data = pd.read_csv(binned_data, 
                       sep = "\t",
                       index_col = 0
                      )
    
    if isinstance(graph['x'], np.ndarray):
        graph['x'] = np.append(graph['x'],
                               data.loc[idxs].values,
                               axis = 1)
    else:
        graph['x'] = data.loc[idxs].values
    
def add_binned_data_to_graphlist(
    graph_list: dict,
    binned_data: str
) -> None:
    for graph in graph_list:
        add_binned_data_to_graph(graph, 
                                 binned_data)

    


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
