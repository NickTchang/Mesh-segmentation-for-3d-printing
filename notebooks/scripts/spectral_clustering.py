import numpy as np
import trimesh as tm
import pandas
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import networkx as nx
import sys
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from scripts import Segmentation_helper as sh
import scipy


def cluster_mesh(mesh, mu = 0.1, n_segments = 2):
    convex_list= mesh.face_adjacency_convex.copy().tolist()
    face_angle = mesh.face_adjacency_angles.copy()
    angular_distance_1 = (1+np.cos(face_angle.round(3).copy())).round(3)
    for i in range(len(convex_list)):
        if convex_list[i]:
            angular_distance_1[i] *= mu
    norm_angular_distance_1 = angular_distance_1/angular_distance_1.mean()
    vector = np.subtract(*mesh.vertices[mesh.face_adjacency_edges.T])
    length = tm.util.row_norm(vector)
    similarity_factor = np.exp(-1*norm_angular_distance_1)
    similarity_factor = (similarity_factor*length).round(3)
    graph = nx.from_edgelist(mesh.face_adjacency)
    edge_weight = dict(zip([tuple(x) for x in mesh.face_adjacency],similarity_factor))
    nx.set_edge_attributes(graph, values = edge_weight, name = 'weight')
    adj_matrix = nx.to_numpy_matrix(graph, nodelist=sorted(graph.nodes()))
    sc = SpectralClustering(n_clusters = n_segments, 
                            affinity='precomputed', 
                            n_init=1, 
                            #verbose=True
                           )
    sc.fit(adj_matrix)
    
    return sc.labels_.copy().tolist()