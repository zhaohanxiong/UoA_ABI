import os
import scipy.io
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from collections import defaultdict

import plotly
import plotly.graph_objects as go

os.chdir("C:/Users/Administrator/Desktop/Export_PVI-11_27_2020-16-11-08_PARTIAL")

def simple_alpha_shape_3D(pos, alpha):
    """
    Compute the alpha shape (concave hull) of a set of 3D points.
    Parameters:
        pos - np.array of shape (n,3) points.
        alpha - alpha value.
    return
        outer surface vertex indices, edge indices, and triangle indices
    """

    tetra = Delaunay(pos)
    # Find radius of the circumsphere.
    # By definition, radius of the sphere fitting inside the tetrahedral needs 
    # to be smaller than alpha value
    # http://mathworld.wolfram.com/Circumsphere.html
    tetrapos = np.take(pos,tetra.vertices,axis=0)
    normsq = np.sum(tetrapos**2,axis=2)[:,:,None]
    ones = np.ones((tetrapos.shape[0],tetrapos.shape[1],1))
    a = np.linalg.det(np.concatenate((tetrapos,ones),axis=2))
    Dx = np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[1,2]],ones),axis=2))
    Dy = -np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[0,2]],ones),axis=2))
    Dz = np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[0,1]],ones),axis=2))
    c = np.linalg.det(np.concatenate((normsq,tetrapos),axis=2))
    r = np.sqrt(Dx**2+Dy**2+Dz**2-4*a*c)/(2*np.abs(a))
    # Find tetrahedrals
    tetras = tetra.vertices[r<alpha,:]
    # triangles
    TriComb = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
    Triangles = tetras[:,TriComb].reshape(-1,3)
    Triangles = np.sort(Triangles,axis=1)
    # Remove triangles that occurs twice, because they are within shapes
    TrianglesDict = defaultdict(int)
    for tri in Triangles: TrianglesDict[tuple(tri)] += 1
    Triangles=np.array([tri for tri in TrianglesDict if TrianglesDict[tri] ==1])
    #edges
    EdgeComb=np.array([(0, 1), (0, 2), (1, 2)])
    Edges=Triangles[:,EdgeComb].reshape(-1,2)
    Edges=np.sort(Edges,axis=1)
    Edges=np.unique(Edges,axis=0)

    Vertices = np.unique(Edges)
    return Vertices,Edges,Triangles


# load data
d_LA = pd.read_csv("Left Atrium.csv").to_numpy()
d_LV = pd.read_csv("LV Myocardium.csv").to_numpy()
d_RA = pd.read_csv("Right Atrium.csv").to_numpy()
d_RV = pd.read_csv("Right Ventricle.csv").to_numpy()
d_AO = pd.read_csv("Aorta.csv").to_numpy()
d_CA = pd.read_csv("Coronary Arteries.csv").to_numpy()
d_CS = pd.read_csv("Coronary Sinus.csv").to_numpy()
d_ES = pd.read_csv("Esophagus.csv").to_numpy()

# compute mesh
_,_,tri_LA = simple_alpha_shape_3D(d_LA[:,1:4], 5)

# plot
fig = go.Figure(data=[go.Mesh3d(x=d_LA[:,0],y=d_LA[:,1],z=d_LA[:,2],
								i=tri_LA[:,0],j=tri_LA[:,1],k=tri_LA[:,2],
								color='red',opacity=0.2,name="Left Atrium")])
								
fig.add_trace(go.Scatter3d(x=d_LV[:,0],y=d_LV[:,1],z=d_LV[:,2],mode='markers',
						   marker=dict(size=2,color="orange",opacity=0.75),name="LV Myocardium"))
fig.add_trace(go.Scatter3d(x=d_RA[:,0],y=d_RA[:,1],z=d_RA[:,2],mode='markers',
						   marker=dict(size=2,color="purple",opacity=0.75),name="Right Atrium"))
fig.add_trace(go.Scatter3d(x=d_RV[:,0],y=d_RV[:,1],z=d_RV[:,2],mode='markers',
						   marker=dict(size=2,color="blue",opacity=0.75),name="Right Ventricles"))
fig.add_trace(go.Scatter3d(x=d_AO[:,0],y=d_AO[:,1],z=d_AO[:,2],mode='markers',
						   marker=dict(size=2,color="yellow",opacity=0.75),name="Aorta"))	
fig.add_trace(go.Scatter3d(x=d_CA[:,0],y=d_CA[:,1],z=d_CA[:,2],mode='markers',
						   marker=dict(size=2,color="brown",opacity=0.75),name="Coronary Arteries"))
fig.add_trace(go.Scatter3d(x=d_CS[:,0],y=d_CS[:,1],z=d_CS[:,2],mode='markers',
						   marker=dict(size=2,color="pink",opacity=0.75),name="Coronary Sinus"))
fig.add_trace(go.Scatter3d(x=d_ES[:,0],y=d_ES[:,1],z=d_ES[:,2],mode='markers',
						   marker=dict(size=2,color="grey",opacity=0.75),name="Esophagus"))

fig.update_layout(margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="white",showlegend=False,
				  scene=dict(aspectmode='manual',aspectratio=dict(x=1,y=1,z=1),
				  #           xaxis=dict(showbackground=False,tickvals=[],ticktext=[],xaxis_title=''),
				  #           yaxis=dict(showbackground=False,tickvals=[],ticktext=[],yaxis_title=''),
				  #           zaxis=dict(showbackground=False,tickvals=[],ticktext=[],zaxis_title='')
							)
				  )
#fig.show()
plotly.offline.plot(fig,'Melb_CARTO.html',auto_open=False)