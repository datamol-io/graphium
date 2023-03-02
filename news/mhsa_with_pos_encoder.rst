**Added:**

* Gaussian Kernel Positional Encoder (gaussian_kernel_pos_encoder.py) processes 3d conformer positioons 
* gaussian kernel embedding can now be pooled together with Laplacian pe and rwse pe

**Changed:**

* updated get_mol_conformer_features() in featurizer.py  to check if positions_3d is used
* note that gaussian kernel encoder takes the entire graph as input different than other pe encoders