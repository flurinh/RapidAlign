Contributing
============

Setting Up Development Environment
-------------------------------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/username/RapidAlign.git
   cd RapidAlign
   
   # Create build directory
   mkdir build && cd build
   
   # Configure for development
   cmake -DCMAKE_BUILD_TYPE=Debug -DRAPID_ALIGN_BUILD_TESTS=ON ..
   
   # Build
   make -j

Code Style
--------

RapidAlign follows these style guidelines:

* C++: Google C++ Style Guide with 2-space indentation
* CUDA: NVIDIA CUDA Programming Guide recommendations
* Python: PEP 8

Running Tests
-----------

.. code-block:: bash

   # Run C++ tests
   cd build
   ctest -V
   
   # Run specific test
   ./test_batch_alignment
   
   # Run Python tests
   cd python
   pytest

Adding New Features
----------------

1. **Create a branch**:

   .. code-block:: bash
   
      git checkout -b feature/your-feature-name

2. **Implementation**:

   * Core CUDA kernels go in `.cu` files
   * C++ API in header files
   * Python bindings in `python/src/pybind.cpp`

3. **Add tests**:

   * Create test case in appropriate test file
   * Include performance benchmark if relevant

4. **Documentation**:

   * Update API reference
   * Add example if needed

5. **Pull Request**:

   * Ensure all tests pass
   * Update documentation
   * Include benchmark results if performance-related

Profiling and Optimization
------------------------

Use NVIDIA profiling tools:

.. code-block:: bash

   # Profile with Nsight Compute
   ncu --set full ./your_test_executable
   
   # Profile with Nsight Systems
   nsys profile ./your_test_executable

Performance tips:

* Minimize device-host memory transfers
* Use grid-based algorithms for large point clouds
* Leverage shared memory for frequently accessed data
* Consider using half precision for appropriate workloads

Reporting Bugs
-----------

Please include:

1. RapidAlign version
2. CUDA version and GPU model
3. Minimal reproducible example
4. Expected vs actual behavior