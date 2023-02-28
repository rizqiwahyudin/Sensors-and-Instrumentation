# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 23:58:17 2023

@author: rizqi
"""

import numpy as np
import math
import matplotlib.pyplot as plt

x = np.arange(0,15,0.5)
y = 1/(1+(2*math.pi*x))

plt.title("Teoretisk Magnitud Respons")
plt.xlabel("Frekvens (Hz)")
plt.ylabel("Magnitud")
plt.plot(x,y)