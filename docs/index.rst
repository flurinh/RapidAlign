.. RapidAlign documentation master file

Welcome to RapidAlign's documentation!
======================================

.. image:: images/logo.svg
   :width: 400
   :alt: RapidAlign Logo

RapidAlign is a high-performance CUDA-accelerated library for fast batch point cloud and graph alignment with deep learning integration.

It provides optimized implementations of point cloud alignment algorithms (Procrustes, ICP, Chamfer) designed specifically for:

- Batch processing of multiple point clouds simultaneously
- Integration with deep learning frameworks (PyTorch)
- Performance optimization through grid acceleration and CUDA streams
- Geometric Graph Neural Network applications

Key Features
------------

- **CUDA-Accelerated**: Leverages GPU compute for fast alignment operations
- **Batched Processing**: Handle multiple point clouds with variable sizes in a single operation
- **Grid-Based Acceleration**: Spatial partitioning for O(n) nearest neighbor search
- **CUDA Streams**: Concurrent processing for optimal GPU utilization
- **PyTorch Integration**: Seamless integration with PyTorch and PyTorch Geometric
- **Production-Ready**: Comprehensive benchmarking and testing tools

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   self
   installation
   getting_started
   reference
   optimizations
   examples
   troubleshooting
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`