import scipy.io
import numpy as np
import os
import plotly
import plotly.graph_objects as go


os.chdir("C:/Users/zxio506/Desktop")

# load data
dat = scipy.io.loadmat("CT.mat")

coords        = dat['CT_coord']
p_conchull    = dat['CT_ashape_x']
tri_conchull  = dat['CT_ashape_tri'].astype(np.int32) - 1 # python is 0 indexed so need to convert this from R
#path          = dat['path']
#path_dist     = dat['pathdist']

# plot actual
if True:
	fig = go.Figure(data=[go.Mesh3d(x=p_conchull[:,0],y=p_conchull[:,1],z=p_conchull[:,2],color='grey',
							        i=tri_conchull[:,0],j=tri_conchull[:,1],k=tri_conchull[:,2],
							        opacity=0.1,name="LA Surface")])
	fig.add_trace(go.Scatter3d(x=coords[:,0],y=coords[:,1],z=coords[:,2],
						       mode='markers',marker=dict(size=1,color="black",opacity=0.1),
						       name="Ensite Generated Mesh"))
	#fig.add_trace(go.Scatter3d(x=path[:,0],y=path[:,1],z=path[:,2],
	#					       mode='markers',marker=dict(size=3,color=path_dist,colorscale='Jet',opacity=0.8),
	#					       name="Catheter Path"))
	fig.update_layout(margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="white",showlegend=False,
					  scene=dict(aspectmode='manual',aspectratio=dict(x=1,y=1,z=1),
					  #           xaxis=dict(showbackground=False,tickvals=[],ticktext=[],xaxis_title=''),
					  #           yaxis=dict(showbackground=False,tickvals=[],ticktext=[],yaxis_title=''),
					  #           zaxis=dict(showbackground=False,tickvals=[],ticktext=[],zaxis_title='')
					            )
					  )
	#fig.show()
	plotly.offline.plot(fig,filename='CT.html',auto_open=False)