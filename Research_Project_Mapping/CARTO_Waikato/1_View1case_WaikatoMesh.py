import os
import plotly
import scipy.io
import numpy as np
import plotly.graph_objects as go

os.chdir("C:/Users/Administrator/Desktop/CatheterPath/2020_Waikato_PointCloud/")
#os.chdir("C:/Users/zxio506/Desktop/Atria_Data/CatheterPath/2020_Waikato_PointCloud/")

n = 5

# patient folders
waikato_patients = ["Patient_0002-Richmond_Michelle",
					"Patient_0003-DEEGAN-Janet",
					"Patient_0004-CRAIG_Leslie",
					"Patient_0005-SIMON_Barry",
					"Patient_0006-CRONIN_Paul",
                                        "Patient_00011-WATKINS_Catheryn"]

# load data
dat = scipy.io.loadmat(waikato_patients[n]+"/catheter.mat")

coords        = dat['coord']
p_conchull    = dat['ashape_x']
tri_conchull  = dat['ashape_tri'].astype(np.int32) - 1 # python is 0 indexed so need to convert this from R
path          = dat['path']
path_dist     = dat['pathdist']

# plot actual
if True:
	fig = go.Figure(data=[go.Mesh3d(x=p_conchull[:,0],y=p_conchull[:,1],z=p_conchull[:,2],color='grey',
							        i=tri_conchull[:,0],j=tri_conchull[:,1],k=tri_conchull[:,2],
							        opacity=0.1,name="LA Surface")])
	fig.add_trace(go.Scatter3d(x=coords[:,0],y=coords[:,1],z=coords[:,2],
						       mode='markers',marker=dict(size=1,color="black",opacity=0.1),
						       name="Ensite Generated Mesh"))
	fig.add_trace(go.Scatter3d(x=path[:,0],y=path[:,1],z=path[:,2],
						       mode='markers',marker=dict(size=3,color=path_dist,colorscale='Jet',opacity=0.8),
							   #mode='markers',marker=dict(size=3,color="blue",opacity=0.8),
						       name="Catheter Path"))
	fig.update_layout(margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="white",showlegend=False,
					  scene=dict(aspectmode='manual',aspectratio=dict(x=1,y=1,z=1),
					             xaxis=dict(showbackground=False,tickvals=[]),
					             yaxis=dict(showbackground=False,tickvals=[]),
					             zaxis=dict(showbackground=False,tickvals=[])
					            )
					  )
	#fig.show()
	plotly.offline.plot(fig,filename="C:/Users/Administrator/Desktop/temp.html",auto_open=False)
	
# plot flipped
if False:
	fig = go.Figure(data=[go.Mesh3d(x=p_conchull[:,1],y=p_conchull[:,0],z=p_conchull[:,2],color='grey',
							        i=tri_conchull[:,1],j=tri_conchull[:,0],k=tri_conchull[:,2],
							        opacity=0.1,name="LA Surface")])
	fig.add_trace(go.Scatter3d(x=coords[:,1],y=coords[:,0],z=coords[:,2],
						       mode='markers',marker=dict(size=1,color="black",opacity=0.1),
						       name="Ensite Generated Mesh"))
	fig.add_trace(go.Scatter3d(x=path[:,1],y=path[:,0],z=path[:,2],
						       mode='markers',marker=dict(size=3,color=path_dist,colorscale='Jet',opacity=0.8),
							   #mode='markers',marker=dict(size=3,color="blue",opacity=0.8),
						       name="Catheter Path"))
	fig.update_layout(margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="white",showlegend=False,
					  scene=dict(aspectmode='manual',aspectratio=dict(x=1,y=1,z=1),
					             xaxis=dict(showbackground=False,tickvals=[]),
					             yaxis=dict(showbackground=False,tickvals=[]),
					             zaxis=dict(showbackground=False,tickvals=[])
					            )
					  )
	#fig.show()
	plotly.offline.plot(fig,filename="C:/Users/Administrator/Desktop/temp.html",auto_open=False)
