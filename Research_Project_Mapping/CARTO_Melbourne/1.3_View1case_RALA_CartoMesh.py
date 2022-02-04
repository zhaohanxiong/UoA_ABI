import os
import plotly
import scipy.io
import numpy as np
import plotly.graph_objects as go

os.chdir("C:/Users/zxio506/Desktop/Atria_Data/CatheterPath/2020_Melbourne_PointCloud/Export_PVI-11_27_2020-16-11-08")

# load CT
def Load_CT_Mesh(f_name):

	dat = scipy.io.loadmat(f_name)

	coords        = dat['coord']
	p_conchull    = dat['ashape_x']
	tri_conchull  = dat['ashape_tri'].astype(np.int32) - 1 # python is 0 indexed so need to convert this from R

	return(coords,p_conchull,tri_conchull)

def Load_CT_Path(f_name):

	dat    = scipy.io.loadmat(f_name)
	coords = dat['coord']
	
	return(coords)
	
# load data
d_LA,p_LA,tri_LA = Load_CT_Mesh("Mesh 1-2-1-ReSRch.mat")
path_LA          = Load_CT_Path("Path 1-2-1-ReSRch.mat")

d_RA,p_RA,tri_RA = Load_CT_Mesh("Mesh 6-RAch.mat")
path_RA          = Load_CT_Path("Path 6-RAch.mat")

# plot
fig = go.Figure(data=[go.Mesh3d(x=p_RA[:,0],y=p_RA[:,1],z=p_RA[:,2],
								i=tri_RA[:,0],j=tri_RA[:,1],k=tri_RA[:,2],
								color='pink',opacity=0.25,name="RA Mesh")])	
fig.add_trace(go.Mesh3d(x=p_LA[:,0],y=p_LA[:,1],z=p_LA[:,2],
						i=tri_LA[:,0],j=tri_LA[:,1],k=tri_LA[:,2],
						color='lightblue',opacity=0.25,name="LA Mesh"))
fig.add_trace(go.Scatter3d(x=path_RA[:,0],y=path_RA[:,1],z=path_RA[:,2],mode='markers',
						   marker=dict(size=2,color="grey",opacity=0.75),name="RA Path"))
fig.add_trace(go.Scatter3d(x=path_LA[:,0],y=path_LA[:,1],z=path_LA[:,2],mode='markers',
						   marker=dict(size=2,color="red",opacity=0.75),name="LA Path"))  
fig.update_layout(margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="white",showlegend=False,
				  scene=dict(aspectmode='manual',aspectratio=dict(x=1,y=1,z=1),
				             xaxis=dict(showbackground=False,tickvals=[],ticktext=[]),
				             yaxis=dict(showbackground=False,tickvals=[],ticktext=[]),
				             zaxis=dict(showbackground=False,tickvals=[],ticktext=[])
							)
				  )

plotly.offline.plot(fig,filename='CatheterPath_LARA.html',auto_open=False)