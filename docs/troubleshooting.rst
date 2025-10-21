Troubleshooting
==============

Common Issues
-----------

CUDA Version Mismatch
~~~~~~~~~~~~~~~~~~

**Issue**: Error about CUDA version incompatibility.

**Solution**: RapidAlign includes a CUDA version detection and compatibility layer. Try:

.. code-block:: bash

   # Force a specific CUDA architecture
   CMAKE_CUDA_ARCHITECTURES=75 cmake ..
   
   # Or specify compatibility mode
   RAPIDALIGN_CUDA_COMPATIBILITY=1 cmake ..

Missing Point Clouds
~~~~~~~~~~~~~~~~

**Issue**: "No points found in point cloud" or similar errors.

**Solution**: 

.. code-block:: cpp

   // Check point cloud size before alignment
   if (source_points.size() == 0 || target_points.size() == 0) {
       std::cerr << "Empty point cloud detected!" << std::endl;
       return;
   }

   // For PyTorch tensors
   if (source.numel() == 0 || target.numel() == 0) {
       print("Empty tensor detected!")
       return

Memory Errors
~~~~~~~~~~

**Issue**: CUDA out of memory errors with large point clouds.

**Solution**: 

1. Try reducing batch size
2. Enable point subsampling:

   .. code-block:: cpp
   
      // Subsample to 1000 points per point cloud
      aligner.setMaxPoints(1000);

3. Use mixed precision:

   .. code-block:: python
   
      # Use half precision for larger batches
      source = source.half()
      target = target.half()
      result = rapidalign.align_point_clouds(source, target)

Poor Alignment Quality
~~~~~~~~~~~~~~~~~~~

**Issue**: Alignment results are not satisfactory.

**Solution**:

1. Initialize with a better starting pose:

   .. code-block:: cpp
      
      // Provide initial transformation guess
      TransformType initial_transform = computeInitialAlignment(source, target);
      TransformType result = aligner.align(source, target, initial_transform);

2. Adjust correspondence threshold:

   .. code-block:: cpp
   
      // More strict correspondence matching
      aligner.setCorrespondenceThreshold(0.05f);  // 5cm threshold

3. Increase iteration count:

   .. code-block:: cpp
   
      // Run more iterations for better convergence
      aligner.setMaxIterations(50);

Debugging Techniques
-----------------

Visualization
~~~~~~~~~~~

Use the included visualization tools:

.. code-block:: bash

   # Visualize alignment results
   ./visualize_graphs --source source.ply --target target.ply --output aligned.ply

Logging
~~~~~

Enable verbose logging:

.. code-block:: cpp

   // Set logging level
   aligner.setVerboseLevel(2);  // 0=none, 1=minimal, 2=detailed

   // For Python
   rapidalign.set_log_level(2)

Benchmarking
~~~~~~~~~~

Identify performance bottlenecks:

.. code-block:: bash

   # Run performance benchmarks
   ./run_tests_and_benchmarks.sh --benchmark-only

   # For Python
   python -m rapidalign.benchmark