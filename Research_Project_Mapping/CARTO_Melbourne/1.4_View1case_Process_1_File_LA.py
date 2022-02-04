import os
import plotly
import scipy.io
import numpy as np
import pandas as pd
import plotly.graph_objects as go

pat_dir = "Export_PVI2-11_27_2021-10-07-14"

#os.chdir("C:/Users/zxio506/Desktop/Atria_Data/CatheterPath/2020_Melbourne_PointCloud/"+pat_dir)
os.chdir("C:/Users/zxio506/Desktop/Export_PVI2-11_27_2021-10-07-14")

def Load_CT_Path(f_name):

	dat    = scipy.io.loadmat(f_name)
	coords = dat['coord']
	
	return(coords)
	
# load data
dat     = scipy.io.loadmat("Mesh 1-1-1-RAFlut2.mat")
d_LA    = dat['coord']
p_LA    = dat['ashape_x']
tri_LA  = dat['ashape_tri'].astype(np.int32) - 1 # python is 0 indexed so need to convert this from R

#path_LA = np.array(pd.read_csv("Path LA.csv"))

#path_LA[:,1] = path_LA[:,1]*-1
#path_LA[:,2] = path_LA[:,2]*-1

#path_LA[:,0] = path_LA[:,0] - 20
#path_LA[:,1] = path_LA[:,1] - 165
#path_LA[:,2] = path_LA[:,2]+1280


# compute path distance
#path_dist = []
#for i in range(path_LA.shape[0]):
#	path_dist.append(np.min(((d_LA[:,0]-path_LA[i,0])**2 + (d_LA[:,1]-path_LA[i,1])**2 + (d_LA[:,2]-path_LA[i,2])**2)**0.5))

# plot
fig = go.Figure(data=[go.Mesh3d(x=p_LA[:,0],y=p_LA[:,1],z=p_LA[:,2],
								i=tri_LA[:,0],j=tri_LA[:,1],k=tri_LA[:,2],
								color='grey',opacity=0.25,name="LA Mesh")])
#fig.add_trace(go.Scatter3d(x=path_LA[:,0],y=path_LA[:,1],z=path_LA[:,2],mode='markers',
#						   marker=dict(size=2,color="red",opacity=0.75),name="LA Path"))
                           #marker=dict(size=2,color=path_dist,colorscale='Jet',opacity=0.5)))
fig.update_layout(margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="white",showlegend=False,
				  scene=dict(aspectmode='manual',aspectratio=dict(x=1,y=1,z=1),
				             #xaxis=dict(showbackground=False,tickvals=[],ticktext=[]),
				             #yaxis=dict(showbackground=False,tickvals=[],ticktext=[]),
				             #zaxis=dict(showbackground=False,tickvals=[],ticktext=[])
							)
				  )
#fig.show()
plotly.offline.plot(fig,filename='CatheterPath_LA.html',auto_open=False)
