# Author: Aaron W. West
# License: MIT
'''
=================================================
Online learning of a dictionary of parts of faces
=================================================
'''
print(__doc__)
import time

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.image import extract_patches_2d

faces = datasets.fetch_olivetti_faces()
print(faces)

