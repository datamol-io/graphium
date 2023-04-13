**Added:**

* Better unit-test coverage for the featurizer, checking the ordering of the nodes, edges, and edge_index

**Changed:**

* All features are now generated as pytorch objects, instead of numpy objects being converted to pytorch
* The `torch_geometric` version is now >= 2.4.0, to allow for the "unbatching" of the sparse tensors, see (PR #7037)[https://github.com/pyg-team/pytorch_geometric/pull/7037]

