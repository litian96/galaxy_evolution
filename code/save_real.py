from __future__ import print_function

import os
import StringIO
import scipy.misc
import numpy as np
from glob import glob
from tqdm import trange
from itertools import chain
from collections import deque
from PIL import Image

from models import *
from utils import save_image


root_path = "./"#self.model_dir

for step in range(3):
	real1_batch, label1_batch = get_image_from_loader()
            #real2_batch, label2_batch = self.get_image_from_loader()
    save_image(real1_batch, os.path.join(root_path, 'test{}_real1.png'.format(step)))