Performance Optimizations
=======================

CUDA Optimizations
----------------

RapidAlign implements several CUDA-specific optimizations:

Grid-Based Acceleration
~~~~~~~~~~~~~~~~~~~~~

For fast nearest neighbor search, RapidAlign uses spatial partitioning:

.. code-block:: cpp

   // Configure grid resolution for spatial partitioning
   aligner.setGridResolution(0.1f);  // Cell size of 0.1 units

CUDA Streams
~~~~~~~~~~~

RapidAlign leverages CUDA streams for concurrent operations:

.. code-block:: cpp

   // Create custom CUDA stream
   cudaStream_t stream;
   cudaStreamCreate(&stream);
   
   // Use stream for concurrent processing
   aligner.setStream(stream);
   
   // Multiple alignments can now run concurrently
   aligner1.align(points1, target1);  // Uses stream 1
   aligner2.align(points2, target2);  // Uses stream 2

Memory Management
~~~~~~~~~~~~~~

Optimized memory patterns reduce allocation overhead:

.. code-block:: cpp

   // Pre-allocate memory for multiple alignments
   aligner.reserveMemory(max_points);
   
   // Reuse memory across multiple calls
   for (int i = 0; i < num_alignments; i++) {
       aligner.align(sources[i], targets[i]);
   }

Batched Kernel Execution
~~~~~~~~~~~~~~~~~~~~~

Processing multiple point clouds in a single kernel:

.. code-block:: python

   # Processes all point clouds in a single kernel launch
   transforms = rapidalign.align_point_clouds(batch_source, batch_target)

Performance Benchmarks
--------------------

.. list-table::
   :header-rows: 1
   
   * - Method
     - Points
     - Batch Size
     - Time (ms)
   * - CPU ICP
     - 10,000
     - 1
     - 250
   * - Basic CUDA
     - 10,000
     - 1
     - 15
   * - RapidAlign
     - 10,000
     - 1
     - 4
   * - RapidAlign
     - 10,000
     - 100
     - 12
   
Memory Usage Optimization
----------------------

Tips for reducing memory footprint:

- Use 16-bit floating point when precision allows
- Enable point cloud downsampling for large inputs
- Adjust grid resolution for specific use cases