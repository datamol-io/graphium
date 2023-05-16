<div align="center">
    <img src="../../docs/images/logo-title.png" height="80px">
    <h3>The Graph Of LIfe Library.</h3>
</div>


## What is in this folder? 

code for base graph layer classes, 
in subfolders there are different GNN layers, positional encoding layers and the graph architecture. 

- ✅ `base_graph_layer.py`: contains `BaseGraphStructure` and `BaseGraphModule` classes, should be inherented by all gnn layers
- ✅ `base_layer.py`: contains base class for different layer, notably `FCLayer` for feedforward layers
- `residual_connections.py`: code for residual connections
- `utils.py`: util for mixed precision


