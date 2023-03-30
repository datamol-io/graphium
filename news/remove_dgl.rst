**Changed:**
* There is less inheritance needed since `DGL` is not supported. So PyTorch Geometric is now used more naturally.

**Removed:**

* Everything related to the `DGL` libabry. DGL is no longer supported.
* Functions like `get_num_nodes` and `get_num_edges` are removed. We can call the method from `pyg` directly. These were just wrappers to allow DGL support.

**Fixed:**

* Minor stuff with the `EncoderManager.max_num_nodes_per_graph` parameter

**Security:**

* <news item>
