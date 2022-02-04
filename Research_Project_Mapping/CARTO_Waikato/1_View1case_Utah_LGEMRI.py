import os
import cv2
import plotly
import numpy as np
from utils import *
import plotly.graph_objects as go

os.chdir("C:/Users/zxio506/Desktop/Atria_Data/Waikato")

waikato_patient = "00003"

# load data
img,lab = np.zeros([640,640,44]),np.zeros([640,640,44])
f1,f2   = os.listdir(waikato_patient+"/lgemri"),os.listdir(waikato_patient+"/label")

for i in range(img.shape[2]):
	img[:,:,i] = cv2.imread(waikato_patient+"/lgemri/"+f1[i],cv2.IMREAD_GRAYSCALE)
	lab[:,:,i] = cv2.imread(waikato_patient+"/label/"+f2[i],cv2.IMREAD_GRAYSCALE)
	
lab = (lab==4).astype(np.int8)

# sample 3D points from the mesh to produce point cloud
def raster2mesh(mask_temp):

	mask = np.copy(mask_temp)

	# interpolate and create a thin outer shell
	mask_temp = np.zeros_like(mask)
	for i in range(mask.shape[2]):
		mask_temp[:,:,i] = cv2.erode(np.uint8(mask[:,:,i]),np.ones((5,5),np.uint8),iterations=2)

	# compute mesh
	x,y,z = np.where((mask - mask_temp) == 1)
	coords = np.array([x,y,z]).T
	vert,edge,triang = simple_alpha_shape_3D(coords,1)
	
	return(coords,triang)

a_coords,a_triang = raster2mesh(lab)

# plot
if True:
	fig = go.Figure(data=[go.Mesh3d(x=a_coords[:,1],y=a_coords[:,0],z=a_coords[:,2],color='grey',
							        i=a_triang[:,1],j=a_triang[:,0],k=a_triang[:,2],
							        opacity=0.1,name="LA Surface")])
	fig.add_trace(go.Scatter3d(x=a_coords[:,1],y=a_coords[:,0],z=a_coords[:,2],
						       mode='markers',marker=dict(size=1,color="black",opacity=0.1)))
	fig.update_layout(margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="white",showlegend=False,
					  scene=dict(aspectmode='manual',aspectratio=dict(x=1,y=1,z=0.75),
					             xaxis=dict(showbackground=False),xaxis_title='', #,tickvals=[],ticktext=[]
					             yaxis=dict(showbackground=False),yaxis_title='', #,tickvals=[],ticktext=[]
					             zaxis=dict(showbackground=False),zaxis_title='',  #,tickvals=[],ticktext=[]
					            )
					  )
	#fig.show()
	
	# save data
	plotly.offline.plot(fig,filename="C:/Users/zxio506/Desktop/temp.html",auto_open=False)
