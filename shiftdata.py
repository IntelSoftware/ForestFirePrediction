from PIL import Image
import os

try:
    os.makedirs('data/shift/train/NoFire')
    os.makedirs('data/shift/val/NoFire')
    os.makedirs('data/shift/train/Fire')
    os.makedirs('data/shift/val/Fire')
except:
    print("folder already exist")

perc = .04
print("percent to change Fire", perc)


def shiftIm(fldrIn, fn, fldrOut, type='png'):
    path = fldrIn + fn
    im = Image.open(path).convert('RGB')

    # Split into 3 channels
    r, g, b = im.split()
    # Increase Reds
    r = r.point(lambda i: i * (1+perc))

    # Decrease Greens
    g = g.point(lambda i: i * (1-perc))

    # Recombine back to RGB image
    result = Image.merge('RGB', (r, g, b))

    result.save(f'{fldrOut}/{fn}')
    im.close()
# cropIm(fldrIn, fn, fldrOut)

import glob
FireList = []
NoFireList = []

fldrIn = "data/output/train/Fire/"
fldrOut = "data/shift/train/Fire"
for fire, fn in enumerate(glob.glob(fldrIn+"*.png")):
    fn_trunc = fn.split('/')[-1]
    FireList.append(fn_trunc)
    shiftIm(fldrIn, fn_trunc, fldrOut, 'png')
print("completed train set")
fldrIn = "data/output/val/Fire/"
fldrOut = "data/shift/val/Fire"
for fire, fn in enumerate(glob.glob(fldrIn+"*.png")):
    fn_trunc = fn.split('/')[-1]
    FireList.append(fn_trunc)
    shiftIm(fldrIn, fn_trunc, fldrOut, 'png')
print("completed val set")
print("done")
