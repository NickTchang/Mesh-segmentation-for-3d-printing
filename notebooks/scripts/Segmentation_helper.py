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
from sklearn.cluster import SpectralClustering
from sklearn import metrics

def plot_mesh(mesh, face_color ,show_normals):
    # set the mesh to to be displayed into viewer_mesh
    viewer_mesh = mesh
    face_normal_vector = viewer_mesh.face_normals
    face_normals = face_normal_vector + viewer_mesh.triangles_center

    #layout
    layout = go.Layout(width=900,
                       height=800,
                      paper_bgcolor='grey',
                      margin=dict(l=20, r=20, t=20, b=20),) 

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

    
def segment_mesh(label, mesh, file_location, fill_holes):
    faces = mesh.faces.copy()
    n_segments = len(np.unique(label))
    submeshes = []
    now = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    mesh_name = f'{mesh=}'.split('=')[0]
    file_path = f'{file_location}/{mesh_name} {now}/'
    
    for i in np.unique(label):
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
        submesh_list.append(tm.Trimesh(vertices=submesh_vert[0], faces=submesh_face[0]))
        
    if(fill_holes):
        for i in range(n_segments):
            tm.repair.fill_holes(submesh_list[i])
            
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    for i in range(n_segments):
        file_name = f'submesh_{i}.stl'
        with open(file_path+file_name,'wb') as f:
            f.write(tm.exchange.stl.export_stl(submesh_list[i]))