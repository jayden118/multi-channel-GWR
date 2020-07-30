#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 21:37:52 2019

@author: chinweihong
"""

from PIL import Image
import glob

files = sorted(glob.glob('snaps/./*.png'))
images = list(map(lambda file: Image.open(file), files))
images[0].save('out.gif', save_all=True, append_images=images[1:], duration=400, loop=0)

files = sorted(glob.glob('dataPlot/./*.png'))
images = list(map(lambda file: Image.open(file), files))
images[0].save('out.gif', save_all=True, append_images=images[1:], duration=200, loop=0)
