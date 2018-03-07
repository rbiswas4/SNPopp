from __future__ import absolute_import, print_function, division

import os

here = __file__
basedir = os.path.split(here)[0]
example_data = os.path.join(basedir, 'example_data')

from .version import __VERSION__ as __version__
from .saltdists import *
from .saltpop import *
