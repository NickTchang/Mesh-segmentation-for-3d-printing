import numpy as np
import trimesh as tm
import networkx as nx
import scipy.sparse

def coarse_segmentation(mesh, n_segments = 2):
    
    # angular distance
    convex_list= mesh.face_adjacency_convex.copy().tolist()
    face_angle = mesh.face_adjacency_angles.copy()
    angular_distance_1 = (1+np.cos(face_angle.round(3).copy())).round(3)
    mu = 0.1
    for i in range(len(convex_list)):
        if convex_list[i]:
            angular_distance_1[i] *= mu
            
    norm_angular_distance_1 = angular_distance_1/angular_distance_1.mean()
    vector = np.subtract(*mesh.vertices[mesh.face_adjacency_edges.T])
    length = tm.util.row_norm(vector)
    similarity_factor = np.exp(-1*norm_angular_distance_1)
    similarity_factor = (similarity_factor*length).round(3)
    
    # adjacency matrix
    graph = nx.from_edgelist(mesh.face_adjacency)
    edge_weight = dict(zip([tuple(x) for x in mesh.face_adjacency],similarity_factor))
    nx.set_edge_attributes(graph, values = edge_weight, name = 'weight')
    
    # laplacian matrix
    lap = nx.linalg.laplacianmatrix.laplacian_matrix(G = graph, nodelist=sorted(graph.nodes()))
    
    # u_snake ausrechnen
    vals, vecs = scipy.sparse.linalg.eigsh(lap, k = 2, which = 'SM')
    vals = vals.round(3)
    vecs = vecs.round(3)
    fiedler_vector = vecs[:,1]
    phi = np.argsort(fiedler_vector)
    phi_inv = np.argsort(phi)
    sorted_fiedler_vector = fiedler_vector[phi]
    first_diff = np.diff(fiedler_vector[phi])
    
    jump_index_tmp = np.argsort(-(first_diff))
    jump_index = jump_index_tmp[:n_segments-1]
    jump_index = jump_index[np.argsort(jump_index)]
    jump_index = jump_index +1 
    jump_index = np.insert(jump_index,0,0)
    jump_index = np.append(jump_index,len(mesh.faces))

    c = np.zeros(n_segments)
    for i in range(n_segments):
        ctmp = np.sum(sorted_fiedler_vector[jump_index[i]:jump_index[i+1]])
        c[i] = ctmp/(jump_index[i+1]-jump_index[i])

    u_snake_sorted = np.repeat(c,np.diff(jump_index))
    u_snake = u_snake_sorted[phi_inv]
    
    return u_snake