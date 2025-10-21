Getting Started
===============

Basic Usage
----------

Using RapidAlign in C++
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   #include <rapidalign/pointcloud_alignment.h>
   
   // Prepare point clouds
   std::vector<float3> source_points, target_points;
   
   // Initialize alignment object
   ra::PointCloudAlignment<float> aligner;
   
   // Perform alignment
   ra::TransformationType transform = aligner.align(source_points, target_points);
   
   // Apply transformation
   auto aligned_points = aligner.transformPointCloud(source_points, transform);

Using RapidAlign in Python
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import rapidalign
   
   # Create point clouds as PyTorch tensors
   source = torch.randn(32, 1000, 3).cuda()  # Batch of 32 point clouds
   target = torch.randn(32, 1000, 3).cuda()
   
   # Perform batch alignment
   transform = rapidalign.align_point_clouds(source, target)
   
   # Apply transformation
   aligned = rapidalign.transform_point_clouds(source, transform)

Batch Processing
--------------

RapidAlign is designed to efficiently handle batches of point clouds:

.. code-block:: python

   # Process 100 point clouds at once
   batch_size = 100
   source_batch = torch.randn(batch_size, 1000, 3).cuda()
   target_batch = torch.randn(batch_size, 1000, 3).cuda()
   
   # One call processes all point clouds
   transformations = rapidalign.align_point_clouds(source_batch, target_batch)

Integration with PyTorch Geometric
-------------------------------

.. code-block:: python

   import torch_geometric as pyg
   import rapidalign.pyg as ra_pyg
   
   # Create PyG graph data
   data = pyg.data.Data(
       x=torch.randn(1000, 16),
       pos=torch.randn(1000, 3),
       edge_index=edge_index
   )
   
   # Use RapidAlign with PyG data
   aligned_data = ra_pyg.align_graph(data, target_graph)