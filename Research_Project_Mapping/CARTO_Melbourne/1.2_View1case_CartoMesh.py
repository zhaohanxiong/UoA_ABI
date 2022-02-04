import os
import plotly
import scipy.io
import numpy as np
import pandas as pd
import plotly.graph_objects as go

os.chdir("C:/Users/Administrator/Desktop/Export_PVI-11_27_2020-16-11-08_PARTIAL")

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
d_LA,p_LA,tri_LA = Load_CT_Mesh("Mesh 1-2-SR.mat")
path_LA          = Load_CT_Path("Path SR.mat")

# plot
fig = go.Figure(data=[go.Mesh3d(x=p_LA[:,0],y=p_LA[:,1],z=p_LA[:,2],
								i=tri_LA[:,0],j=tri_LA[:,1],k=tri_LA[:,2],
								color='grey',opacity=0.2,name="Mesh")])								
fig.add_trace(go.Scatter3d(x=path_LA[:,0],y=path_LA[:,1],z=path_LA[:,2],mode='markers',
						   marker=dict(size=2,color="red",opacity=0.75),name="Path"))
fig.update_layout(margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="white",showlegend=False,
				  scene=dict(aspectmode='manual',aspectratio=dict(x=1,y=1,z=1),
				  #           xaxis=dict(showbackground=False,tickvals=[],ticktext=[],xaxis_title=''),
				  #           yaxis=dict(showbackground=False,tickvals=[],ticktext=[],yaxis_title=''),
				  #           zaxis=dict(showbackground=False,tickvals=[],ticktext=[],zaxis_title='')
							)
				  )
#fig.show()
plotly.offline.plot(fig,filename='SR.html',auto_open=False)