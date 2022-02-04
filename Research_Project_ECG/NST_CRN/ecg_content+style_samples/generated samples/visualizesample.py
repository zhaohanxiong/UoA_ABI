import scipy.io
import matplotlib.pyplot as plt

dat = scipy.io.loadmat("sample4_A")

plt.subplot(3, 1, 1)
plt.title('Content')
plt.plot(dat["content"][0,:])
plt.subplot(3, 1, 2)
plt.title('Style')
plt.plot(dat["style"][0,:])
plt.subplot(3, 1, 3)
plt.title('Generated')
plt.plot(dat["generate"][0,:])
plt.show()
