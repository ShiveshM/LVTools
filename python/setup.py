
from distutils.core import setup
from distutils.extension import Extension
import os.path
import sys
import numpy

if sys.platform == 'win32' or sys.platform == 'win64':
    print 'Windows is not a supported platform.'
    quit()

else:
    include_dirs = ['/data/icecube/software/LVTools_package/anaconda/envs/lvtools/include/',
                    '/data/icecube/software/LVTools_package/nuSQuIDS/include',
                    '/data/icecube/software/LVTools_package/SQuIDS/include',
                    '/data/icecube/software/dependencies/boost_1_61_0/include',
                    '/data/icecube/software/dependencies/gsl-2.2/include',
                    '/data/icecube/software/LVTools_package/PhysTools/include',
                    numpy.get_include(),
                    '../inc/',
                    '.']
    libraries = ['python2.7','boost_python', 'PhysTools',
                 'SQuIDS','nuSQuIDS',
                 'gsl','gslcblas','m',
                 'hdf5','hdf5_hl','PhysTools']

    if sys.platform.startswith('linux'):
      libraries.append('cxxrt')

    library_dirs = ['/data/icecube/software/LVTools_package/anaconda/envs/lvtools/lib',
                    '/data/icecube/software/LVTools_package/nuSQuIDS/lib',
                    '/data/icecube/software/LVTools_package/SQuIDS/lib',
                    '/data/icecube/software/dependencies/boost_1_61_0/lib',
                    '/data/icecube/software/LVTools_package/libcxxrt/build/lib',
                    '/data/icecube/software/LVTools_package/PhysTools/lib',
                    '/data/icecube/software/dependencies/gsl-2.2/lib',]

files = ['lvsearchpy.cpp']

setup(name = 'lvsearchpy',
      ext_modules = [
          Extension('lvsearchpy',files,
              library_dirs=library_dirs,
              libraries=libraries,
              include_dirs=include_dirs,
              extra_objects=["../mains/lbfgsb.o","../mains/linpack.o"],
              extra_compile_args=['-O3','-fPIC','-std=c++11','-Wno-unused-local-typedef'],
              depends=[]),
          ]
      )

