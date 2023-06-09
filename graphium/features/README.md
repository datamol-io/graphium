<div align="center">
    <img src="../../docs/images/logo-title.png" height="80px">
    <h3>The Graph Of LIfe Library.</h3>
</div>


## What is in this folder? 

- ✅ `featurizer.py`: featurization code for the molecules, adding node, edge and graph features to the mol object
- `nmp.py`: check if a string can be converted to float, helper function for featurization
- `positional_encoding.py`: code for computing all raw positional and structural encoding of the graph, see `graph_positional_encoder` function
- `properties.py`: code for computing properties of the molecule
- `rw.py`: code for computing random walk positional encoding
- `spectral.py`: code for computing the spectral positional encoding such as the Laplacian eigenvalues and eigenvectors