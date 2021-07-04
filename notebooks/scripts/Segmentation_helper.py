import numpy as np
import trimesh as tm
import pandas
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import networkx as nx
import sys
import os
import datetime
import random
from sklearn.cluster import SpectralClustering
from sklearn import metrics

def plot_mesh(mesh, face_color = [], show_normals = False):
    # set the mesh to to be displayed into viewer_mesh
    viewer_mesh = mesh
    face_normal_vector = viewer_mesh.face_normals
    face_normals = face_normal_vector + viewer_mesh.triangles_center

    maximum = float(mesh.vertices.max())+1
    minimum = float(mesh.vertices.min())-1
    #layout
    layout = go.Layout(width=1000,
                       height=1000,
                       paper_bgcolor='grey',
                       margin=dict(l=20, r=20, t=20, b=20),
                       scene_aspectmode='cube',
                       scene = dict(
                                    xaxis = dict(range = [minimum,maximum]),
                                    yaxis = dict(range = [minimum,maximum]),
                                    zaxis = dict(range = [minimum,maximum]),
                                    aspectmode='manual', #this string can be 'data', 'cube', 'auto', 'manual'
                                    #a custom aspectratio is defined as follows:
                                    # mit dict(x=1, y=1, z=1) ist aequivalent zu aspectmode = cube
                                    aspectratio=dict(x=1, y=1, z=1)
                                    )
                       )

    vertices = viewer_mesh.vertices
    faces = viewer_mesh.faces
    x, y, z = vertices[:,:3].T
    I, J, K = faces.T

    plot_mesh = go.Mesh3d(
                x=-x,
                y=z,
                z=y,
                #vertexcolor=vertices[:, 3:], #the color codes must be triplets of floats  in [0,1]!!                      
                i=I,
                j=J,
                k=K,
                name='',
                showscale=False,
                facecolor = face_color)

    if(show_normals):
        # scatterplot for the face normals
        x = face_normals[:,:1].copy().reshape(face_normals[:,:1].copy().shape[0],)
        y = face_normals[:,1:2].copy().reshape(face_normals[:,:1].copy().shape[0],)
        z = face_normals[:,2:3].copy().reshape(face_normals[:,:1].copy().shape[0],)
        normals = go.Scatter3d(x=-x, y=z, z=y,mode='markers', marker = dict(color = 'green'))
        fig1 = go.Figure(data=[plot_mesh,normals], layout=layout)
    else:
        fig1 = go.Figure(data=plot_mesh, layout=layout)
        
    fig1.show()
    
    
def plot_graph(mesh):
    graph = nx.from_edgelist(mesh.face_adjacency)
    pos=nx.spring_layout(graph)
    plt.figure(figsize=(40,30))
    nx.draw_networkx_nodes(graph,pos,node_size=300)                 # draw nodes
    nx.draw_networkx_edges(graph,pos,edgelist=mesh.face_adjacency,width=2) # draw edges

    
def segment_mesh(label, mesh, fill_holes = False, save_mesh = False, file_location = ''):
    # label ist eine list von integers, die die faces angibt, bsp.: [0,1,1,0,2,2,1,0]
    faces = mesh.faces.copy()
    n_segments = len(np.unique(label))
    submeshes = []

    
    for i in range(n_segments):
        mask = np.equal(label, i)
        submeshes.append(faces[mask].copy())
    
    submesh_vert = []
    for i in range(n_segments):
        submesh_vert.append(mesh.vertices[np.unique(submeshes[i])].tolist())
    
    submesh_face = submeshes.copy()
    for k in range(n_segments):
        uni_num = np.unique(submesh_face[k])
        for i in range(len(uni_num)):
            submesh_face[k] = np.where(submesh_face[k] == uni_num[i],i,submesh_face[k])
            
    submesh_list = []
    for i in range(n_segments):
        submesh_list.append(tm.Trimesh(vertices=submesh_vert[i], faces=submesh_face[i]))
        
    if(fill_holes):
        for i in range(n_segments):
            tm.repair.fill_holes(submesh_list[i])
            
    if (save_mesh):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        mesh_name = f'{mesh=}'.split('=')[0]
        file_path = f'{file_location}/{mesh_name} {now}/'
        
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        for i in range(n_segments):
            file_name = f'submesh_{i}.stl'
            with open(file_path+file_name,'wb') as f:
                f.write(tm.exchange.stl.export_stl(submesh_list[i]))
                
    return submesh_list


def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    Licensed under Creative Commons Attribution-Share Alike 
    
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    G: the graph (must be a tree)
    
    root: the root node of current branch 
    - if the tree is directed and this is not given, 
      the root will be found and used
    - if the tree is directed and this is given, then 
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, 
      then a random choice will be used.
    
    width: horizontal space allocated for this branch - avoids overlap with other branches
    
    vert_gap: gap between levels of hierarchy
    
    vert_loc: vertical location of root
    
    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''
    
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

            
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)