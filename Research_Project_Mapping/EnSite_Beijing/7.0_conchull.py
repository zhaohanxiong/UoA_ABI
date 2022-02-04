from utils import *
import scipy.io
import numpy as np
import plotly.graph_objects as go

# load data
dat = scipy.io.loadmat("temp.mat")

p_chull   = dat['ashape_vert']

_,_,triang = simple_alpha_shape_3D(p_chull,5)

if True:

	fig = go.Figure(data=[go.Mesh3d(x=p_chull[:,1],y=p_chull[:,0],z=p_chull[:,2],color='grey',
							        i=triang[:,1],j=triang[:,0],k=triang[:,2],
							        opacity=0.2,name="LA Surface")])
	fig.add_trace(go.Scatter3d(x=p_chull[:,1],y=p_chull[:,0],z=p_chull[:,2],
						       mode='markers',marker=dict(size=2,color="black",opacity=0.25)))
	fig.update_layout(margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="white",showlegend=False,
					  scene=dict(aspectmode='manual',aspectratio=dict(x=1,y=1,z=0.75),
					             xaxis=dict(showbackground=False),xaxis_title='', #,tickvals=[],ticktext=[]
					             yaxis=dict(showbackground=False),yaxis_title='', #,tickvals=[],ticktext=[]
					             zaxis=dict(showbackground=False),zaxis_title='',  #,tickvals=[],ticktext=[]
					            )
					  )
	fig.show()
