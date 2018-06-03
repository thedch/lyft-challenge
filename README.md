# Lyft Challenge Writeup

This segmentation algorithm was developed based on the carvana network from Jeremy Howard.

I use a U-Net style architecture, with resnet 34 for feature extraction and then several conv transpose layers to upsample the image. 