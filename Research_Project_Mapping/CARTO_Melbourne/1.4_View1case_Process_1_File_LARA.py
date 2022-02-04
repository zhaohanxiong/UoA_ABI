import os
import plotly
import scipy.io
import numpy as np
import pandas as pd
import plotly.graph_objects as go

os.chdir("C:/Users/zxio506/Desktop/Atria_Data/CatheterPath/2020_Melbourne_PointCloud/Export_AT-08_08_2017-13-26-43")

def Load_CT_Path(f_name):

	dat    = scipy.io.loadmat(f_name)
	coords = dat['coord']
	
	return(coords)
	
# load data
dat     = scipy.io.loadmat("Mesh LA.mat")
d_LA    = dat['coord']
p_LA    = dat['ashape_x']
tri_LA  = dat['ashape_tri'].astype(np.int32) - 1 # python is 0 indexed so need to convert this from R

dat     = scipy.io.loadmat("Mesh RA.mat")
d_RA    = dat['coord']
p_RA    = dat['ashape_x']
tri_RA  = dat['ashape_tri'].astype(np.int32) - 1 # python is 0 indexed so need to convert this from R

path_LA = np.array(pd.read_csv("Path LA.csv"))
path_RA = np.array(pd.read_csv("Path RA.csv"))

# plot
fig = go.Figure(data=[go.Mesh3d(x=p_LA[:,0],y=p_LA[:,1],z=p_LA[:,2],
								i=tri_LA[:,0],j=tri_LA[:,1],k=tri_LA[:,2],
								color='grey',opacity=0.25,name="LA Mesh")])
fig.add_trace(go.Scatter3d(x=path_LA[:,0],y=path_LA[:,1],z=path_LA[:,2],mode='markers',
						   marker=dict(size=2,color="red",opacity=0.75),name="LA Path")) 
fig.add_trace(go.Mesh3d(x=p_RA[:,0],y=p_RA[:,1],z=p_RA[:,2],
						i=tri_RA[:,0],j=tri_RA[:,1],k=tri_RA[:,2],
						color='purple',opacity=0.25,name="RA Mesh"))
fig.add_trace(go.Scatter3d(x=path_RA[:,0],y=path_RA[:,1],z=path_RA[:,2],mode='markers',
						   marker=dict(size=2,color="yellow",opacity=0.5),name="RA Path"))
fig.update_layout(margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="white",showlegend=False,
				  scene=dict(aspectmode='manual',aspectratio=dict(x=1,y=1,z=1),
				             xaxis=dict(showbackground=False,tickvals=[],ticktext=[]),
				             yaxis=dict(showbackground=False,tickvals=[],ticktext=[]),
				             zaxis=dict(showbackground=False,tickvals=[],ticktext=[])
							)
				  )

plotly.offline.plot(fig,filename='CatheterPath_LARA.html',auto_open=False)
