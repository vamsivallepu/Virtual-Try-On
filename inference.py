from Model import Model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import cv2

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model = Model("checkpoints/jpp.pb",
              "checkpoints/gmm.pth", 
              "checkpoints/tom.pth")
img = np.array(Image.open("data/test3.jpg"))
c_img = np.array(Image.open("data/test4.jpg"))
start = time.time()
result,trusts = model.predict(img, c_img, need_pre=False,check_dirty=True)
if result is not None:
    end = time.time()
    print("time:"+str(end-start))
    print("Confidence:"+str(trusts))
    cv2.imwrite("result.jpg",result)