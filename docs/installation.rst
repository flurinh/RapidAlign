Installation
============

Prerequisites
------------

* CUDA toolkit 11.0+
* C++17 compatible compiler
* CMake 3.18+
* Python 3.8+ (for Python bindings)
* PyTorch 1.9+ (for PyTorch integration)

From Source
----------

.. code-block:: bash

   git clone https://github.com/username/RapidAlign.git
   cd RapidAlign
   mkdir build && cd build
   cmake ..
   make -j
   make install

Using pip (Python bindings)
--------------------------

.. code-block:: bash

   pip install RapidAlign

Using conda
----------

.. code-block:: bash

   conda install -c conda-forge rapidalign

Cross-version CUDA Compatibility
-------------------------------

RapidAlign features automatic CUDA version detection and compatibility
handling for mixed CUDA environments.

.. code-block:: bash

   # Specify a different CUDA version than your system default
   CMAKE_CUDA_ARCHITECTURES=75 cmake ..