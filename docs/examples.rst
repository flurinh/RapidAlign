Examples
========

Basic Point Cloud Alignment
-------------------------

.. code-block:: python

   import torch
   import rapidalign
   
   # Generate random point clouds
   source = torch.randn(1000, 3).cuda()
   target = torch.randn(1000, 3).cuda()
   
   # Align point clouds
   transform = rapidalign.align_point_clouds(source, target)
   
   # Apply transformation
   aligned = rapidalign.transform_point_clouds(source, transform)
   
   # Calculate alignment error
   error = torch.mean(torch.norm(aligned - target, dim=1))
   print(f"Alignment error: {error:.4f}")

PyTorch Integration with Gradients
-------------------------------

.. code-block:: python

   import torch
   import rapidalign
   
   # Create point clouds that require gradients
   source = torch.randn(1000, 3, requires_grad=True).cuda()
   target = torch.randn(1000, 3).cuda()
   
   # Create a network that produces point offsets
   class PointNet(torch.nn.Module):
       def __init__(self):
           super().__init__()
           self.mlp = torch.nn.Sequential(
               torch.nn.Linear(3, 64),
               torch.nn.ReLU(),
               torch.nn.Linear(64, 3)
           )
       
       def forward(self, x):
           return x + self.mlp(x)
   
   model = PointNet().cuda()
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   
   # Training loop
   for epoch in range(100):
       optimizer.zero_grad()
       
       # Generate point cloud from network
       predicted_points = model(source)
       
       # Align with target
       transform = rapidalign.align_point_clouds(predicted_points, target)
       aligned = rapidalign.transform_point_clouds(predicted_points, transform)
       
       # Compute loss
       loss = torch.mean(torch.norm(aligned - target, dim=1))
       
       # Backpropagate
       loss.backward()
       optimizer.step()
       
       if epoch % 10 == 0:
           print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

Graph Alignment with PyTorch Geometric
-----------------------------------

.. code-block:: python

   import torch
   import torch_geometric as pyg
   import rapidalign.pyg as ra_pyg
   
   # Create source and target graphs
   source_pos = torch.randn(100, 3).cuda()
   source_edge_index = pyg.nn.radius_graph(source_pos, r=0.2)
   
   target_pos = torch.randn(100, 3).cuda()
   target_edge_index = pyg.nn.radius_graph(target_pos, r=0.2)
   
   source_graph = pyg.data.Data(pos=source_pos, edge_index=source_edge_index)
   target_graph = pyg.data.Data(pos=target_pos, edge_index=target_edge_index)
   
   # Align graphs
   aligned_graph = ra_pyg.align_graph(source_graph, target_graph)
   
   # Calculate alignment error
   error = torch.mean(torch.norm(aligned_graph.pos - target_graph.pos, dim=1))
   print(f"Graph alignment error: {error:.4f}")