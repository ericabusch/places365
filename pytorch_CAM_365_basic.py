# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception
# modifications made for places 365 resnet18
import io
import requests
from PIL import Image
import torch as torch
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import pdb
import os
from datetime import datetime as dt

# input image
# I made changes here
LABELS_URL = '/Users/ericabusch/Desktop/Thesis/getting_started/places365/IO_places365.txt'
IMG_URL = '/Users/ericabusch/Desktop/Thesis/getting_started/sample_data/sample_equirect.jpg'

# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
# model_id = 1
# if model_id == 1:
#     net = models.squeezenet1_1(pretrained=True)
#     finalconv_name = 'features' # this is the last conv layer of the network
# elif model_id == 2:
#     net = models.resnet18(pretrained=True)
#     finalconv_name = 'layer4'
# elif model_id == 3:
#     net = models.densenet161(pretrained=True)
#     finalconv_name = 'features'

# using resnet18 pretrained on places365
arch = 'resnet18'
path2model = '/Users/ericabusch/Desktop/Thesis/getting_started/models'
# load the pre-trained weights
model_file = '%s/%s_places365.pth.tar' %path2model %arch
# if not os.access(model_file, os.W_OK):
#     weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
#     os.system('wget ' + weight_url)
net = models.__dict__[arch](num_classes=365)
import torch.load
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
net.load_state_dict(state_dict)
net.eval()
finalconv_name = 'layer4'


# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

net._modules.get(finalconv_name).register_forward_hook(hook_feature)

# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

response = requests.get(IMG_URL)
img_pil = Image.open(io.BytesIO(response.content))
img_pil.save('test.jpg')

img_tensor = preprocess(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))
logit = net(img_variable)

# download the imagenet category list
classes = {int(key):value for (key, value)
          in requests.get(LABELS_URL).json().items()}

h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.numpy()
idx = idx.numpy()

# output the prediction
for i in range(0, 5):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

# generate class activation mapping for the top1 prediction
CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

# render the CAM and output
now = dt.now()
outfile = now.strtftime("%m%d%Y-%H%M%S.jpg")
print('output %s for the top1 prediction: %s' %outfile %classes[idx[0]])
img = cv2.imread('test.jpg')
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite('%s' %outfile, result)

