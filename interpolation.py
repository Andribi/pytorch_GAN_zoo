# USAGE
# python analyze.py
# import the necessary packages
#from pyimagesearch import config
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch
from models.progressive_gan import ProgressiveGAN
from models.utils.utils import getLastCheckPoint



l = []
at_sample = [0, 0, 1, 0, 0, 1, 0, 1, 0, 0]
for at in range(16):
    l.append(at_sample)


att_v = torch.tensor(l)




def interpolate(n):
    # sample the two noise vectors z1 and z2
    (noise, labels) = model.buildNoiseData(16, att_v)

    # define the step size and sample numbers in the range (0, 1) at
    # step intervals
    step = 1 / n
    lam = list(np.arange(0, 1, step))

    # initialize a tensor for storing interpolated images
    interpolatedImages = torch.zeros([n, 3, 256, 256])
    # iterate over each value of lam
    for i in range(n):
        # compute interpolated z
        zInt = (1 - lam[i]) * noise[0] + lam[i] * noise[1]

        # generate the corresponding in the images space
        with torch.no_grad():
            outputImage = model.test(zInt.reshape(-1, 532))
            interpolatedImages[i] = outputImage
    # return the interpolated images
    return interpolatedImages


# load the pre-trained PGAN model

checkpoint = getLastCheckPoint('D:\\PyCharmProjects\\pytorch_GAN_zoo\\checkpoints\\utkface_v1', 'utkface_v1')
trainConfig, pathModel, pathTmpData = checkpoint





'''GANTrainer = trainerModule(pathDB,
                           useGPU=True,
                           visualisation=vis_module,
                           lossIterEvaluation=kwargs["evalIter"],
                           checkPointDir=checkPointDir,
                           saveIter=kwargs["saveIter"],
                           modelLabel=modelLabel,
                           partitionValue=partitionValue,
                           **trainingConfig)
'''

#GANTrainer.loadSavedTraining(pathModel, trainConfig, pathTmpData)


model = ProgressiveGAN()
model.load(pathModel)

#model = torch.load('checkpoints\\utkface_vis\\utkface_vis_s6_i200000.pt')

#model = torch.hub.load("facebookresearch/pytorch_GAN_zoo:hub",
#                       "PGAN", model_name="celebAHQ-512", pretrained=True, useGPU=True)
# call the interpolate function
interpolatedImages = interpolate(16)
# visualize output images
grid = torchvision.utils.make_grid(
    interpolatedImages.clamp(min=-1, max=1), scale_each=True,
    normalize=True)
plt.figure(figsize=(10, 10))
plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
plt.show()
# save visualizations
#torchvision.utils.save_image(interpolatedImages.clamp(min=-1, max=1),
#                             config.INTERPOLATE_PLOT_PATH, nrow=config.NUM_IMAGES,
#                             scale_each=True, normalize=True)
