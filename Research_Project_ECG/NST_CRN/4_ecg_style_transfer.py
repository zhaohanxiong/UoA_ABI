import tensorflow as tf
import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from biosppy.signals import ecg

#STYLE_FILENAME 	 = "ecg_content+style_samples/style/N1.mat"
STYLE_FILENAME   = "C:/Users/Administrator/Desktop/cinc2017/dataset/A00002.mat"
CONTENT_FILENAME = "ecg_content+style_samples/content/N_fake60bpm.mat"
OUTPUT_FILENAME	 = "sample6_A"

N_FFT,W_LEN,seed = 64,64,1111

np.random.seed(seed)



### load files
content_sig = scipy.io.loadmat(CONTENT_FILENAME)['val'][:,0]
#style_sig = scipy.io.loadmat(STYLE_FILENAME)['val'][:,0]
style_sig = scipy.io.loadmat(STYLE_FILENAME)['val'][0,0:2500]
style_sig = style_sig - np.min(style_sig)
style_sig = style_sig/np.max(style_sig)*255
#plt.plot(style_sig);plt.show();sys.exit()

a_content 	= np.log1p(np.abs(librosa.stft(content_sig,N_FFT,win_length=W_LEN)))
a_style 	= np.log1p(np.abs(librosa.stft(style_sig,N_FFT,win_length=W_LEN)))

N_CHANNELS 	= a_content.shape[0]
N_SAMPLES 	= a_content.shape[1]

a_style 	= a_style[:N_CHANNELS,:N_SAMPLES]



### Compute content and style feats
def conv_2d(incoming,kernel):

	kernel_tf = tf.constant(kernel, dtype='float32')
	conv = tf.nn.conv2d(incoming, kernel_tf, strides=[1,1,1,1], padding="SAME")
	net = tf.nn.relu(conv)
	
	return(net)

N_FILTERS = 1024

a_content_tf = np.ascontiguousarray(a_content.T[None,None,:,:])
a_style_tf   = np.ascontiguousarray(a_style.T[None,None,:,:])

# filter shape is [filter_height, filter_width, in_channels, out_channels]
kernel1 = np.random.randn(1, 16, N_CHANNELS, N_FILTERS) * np.sqrt(2) * np.sqrt(2.0 / ((N_CHANNELS + N_FILTERS) * 16))

g = tf.Graph()
with g.as_default(), g.device('/gpu:0'), tf.Session() as sess:

	# data shape is [batch, in_height, in_width, in_channels]
	x = tf.placeholder('float32', [1,1,N_SAMPLES,N_CHANNELS], name="x")
	
	net1 = conv_2d(x,kernel1)

	content_features = net1.eval(feed_dict={x: a_content_tf})
	
	features1   = np.reshape(net1.eval(feed_dict={x: a_style_tf}), (-1,N_FILTERS))
	style_gram1 = np.matmul(features1.T, features1) / N_SAMPLES

ALPHA = 0.1 # larger weights content more, smaller weights style more

with tf.Graph().as_default():

	# Build graph with variable input
	tf.set_random_seed(seed)
	x = tf.Variable(np.random.randn(1,1,N_SAMPLES,N_CHANNELS).astype(np.float32)*1e-3, name="x")

	net1 = conv_2d(x,kernel1)

	content_loss = ALPHA * 2 * tf.nn.l2_loss(net1 - content_features)

	feats1 = tf.reshape(net1, (-1, int(net1.get_shape()[3])))
	gram1  = tf.matmul(tf.transpose(feats1), feats1) / N_SAMPLES

	style_loss = 2 * tf.nn.l2_loss(gram1 - style_gram1)

	# Overall loss
	loss = content_loss + style_loss
	opt = tf.contrib.opt.ScipyOptimizerInterface(loss, method='L-BFGS-B', options={'maxiter': 250})

	# Optimization
	with tf.Session() as sess:
	
		sess.run(tf.global_variables_initializer())
	   
		print('Started optimization...')
		opt.minimize(sess)

		print('Final loss:', loss.eval())
		result = x.eval()



### Invert spectrogram and save the result
a = np.zeros_like(a_content)
a[:N_CHANNELS,:] = np.exp(result[0,0].T) - 1

# phase reconstruction
p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
for i in range(100):
	S = a * np.exp(1j*p)
	output = librosa.istft(S)
	p = np.angle(librosa.stft(output, N_FFT))

#scipy.io.savemat("ecg_content+style_samples/generated samples/out",mdict={'val':output})



### Post process ecg, flip QRS peaks, and filter signal
r_peak_pos = ecg.ecg(signal=content_sig, sampling_rate=300, show=False)[2]

for r in r_peak_pos:
	if output[r] < output[r-10] and output[r] < output[r+10]:
		base = np.mean([output[r+10],output[r-10]])
		output[(r-10):(r+10)] += 2*(base - output[(r-10):(r+10)])

filtered_output = ecg.ecg(signal=output, sampling_rate=300, show=False)[1]
style_sig		= ecg.ecg(signal=style_sig, sampling_rate=300, show=False)[1]



### Plot generated ecg
def plot_3_sig(dat1,dat2,dat3):
	plt.subplot(3, 1, 1)
	plt.title('Content')
	plt.plot(dat1)
	plt.subplot(3, 1, 2)
	plt.title('Style')
	plt.plot(dat2)
	plt.subplot(3, 1, 3)
	plt.title('Generated')
	plt.plot(dat3)
	plt.show()
	
def plot_sig(dat):
	plt.plot(dat)
	plt.show()

plot_3_sig(content_sig,style_sig,filtered_output)



### Save output
scipy.io.savemat("C:/Users/Administrator/Desktop/"+OUTPUT_FILENAME+".mat",
					mdict={'content':content_sig,'style':style_sig,'generate':filtered_output})
