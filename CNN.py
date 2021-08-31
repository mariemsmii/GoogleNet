import caffe
import sys
mb=sys.argv[1]
nb=sys.argv[2]
myImage=[]
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

import sys
caffe_root = '/home/mariem/Desktop/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')



import os
#caffe_root = '/home/mariem/Desktop/caffe/'
if os.path.isfile(caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'):
    print ('GoogleNet found.')
else:
    print ('Downloading pre-trained GoogleNet model...')

caffe.set_mode_cpu()

model_def = caffe_root + 'models/bvlc_googlenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't p subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print ('mean-subtracted values:', zip('BGR', mu))

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channe

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
#net.blobs['data'].reshape(50,        # batch size
 #                         3,         # 3-channel (BGR) images
                       #  227, 227)  # image size is 227x227


print(mb)
pp=0
m=0
for j in range (m,m+int(mb)):
   n=0  
   print(j) 
   for i in range (n,n+int(nb)):  
      myImage.append('/examples/database/'+str(j)+'/'+str(i)+'.jpg') #put your database' name
      print (myImage[pp])

      image = caffe.io.load_image(caffe_root +myImage[pp])
      transformed_image = transformer.preprocess('data', image)
      pp=pp+1
      net.blobs['data'].data[...]=transformed_image
      output = net.forward()
      output_prob = output['prob'][0]
      print ("predicted class is:",output_prob.argmax()) 
      labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
      labels = np.loadtxt(labels_file, str, delimiter='\t')
      print ('output label:', labels[output_prob.argmax()])
      top_inds = output_prob.argsort()[::-1][:5] 
      print ('probabilities and labels:')
      print ((output_prob[top_inds], labels[top_inds]))
      n=n+1
m=m+1







