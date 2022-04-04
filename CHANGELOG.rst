=====================
goli Change Log
=====================

.. current developments

v0.2.12
====================

**Added:**

* New functions use specifically to profile parts of the code.
* Support of Pytorch >=1.9 with a single line fixed from torch.tensor import Tensor -> from torch import Tensor.

**Changed:**

* Featurization of molecules into graphs is 3 times faster on Laptop and 2.2 times faster on GCP instance.
  * Improvements in atom properties speed by getting atomic information from global lists, not from a dataframe where indexing is slow.
  * Improvements in bond properties speed by getting atomic information from global lists, not from a dataframe. Also optimized the get_estimated_bond_length.
  * Improved positional encodings speed by using scipy.linalg.eig instead of scipy.linalg.eigs. The second is faster for large graphs, but not small ones. Also some code optimization. There might be tiny differences in eigenvectors, but it shouldn't change much the network's output. For most molecules, the eigenvectors are identical.
  * Other optimizations using more native python. Less numpy, more lists.
  * Moved back to loky for parallel backend. It benefits more from the optimized code, and memory leaks seem to no longer be an issue. Probably since the code is closer to native python with numpy arrays being replaced by lists.
  * Checked that the featurization gives the same results as before.

**Deprecated:**

* Nothing

**Removed:**

* Nothing

**Fixed:**

* Nothing

**Security:**

* Nothing

**Authors:**

* Dom
* Hadrien Mary



v0.2.11
====================

**Added:**

* Nothing

**Changed:**

* Nothing

**Deprecated:**

* Nothing

**Removed:**

* Nothing

**Fixed:**

* Fix the lr-scheduler that did not use the `monitor` and `mode` correctly.

**Security:**

* Nothing

**Authors:**

* Dom
* Hadrien Mary



v0.2.10
====================

**Authors:**

* Dom



v0.2.9
====================

**Authors:**

* Dom
* Hadrien Mary



v0.2.8
====================

**Added:**

* Add FLAG (Free Large-Scale Adversarial Augmentation on Graphs), a form of adversarial data augmentation to boost performance of GNNs across various molecular tasks. FLAG accomplishes this by iteratively augmenting node features with gradient-based adversarial perturbations during training. See the paper on `arXiv <https://arxiv.org/abs/2010.09891>`_ and authors' code on `GitHub <https://github.com/devnkong/FLAG>`_ for more information.

**Authors:**

* Dom
* Gabriela Moisescu-Pareja



v0.2.7
====================

**Authors:**

* Dom



v0.2.5
====================

**Added:**

* Caching of the pre-processed dataset when loading from a zip file or from the cloud. These operations used to be extremely slow.

**Changed:**

* Avoid memory leak from joblib by using the multiprocessing backend instead of loky
* Load dictionaries of arrays instead of DGLGraph. This speeds up the pre-processing significantly, with only minor slow-down to build the DGL graphs during training.
* Use numpy arrays instead of torch Tensors when creating the dict of graphs. Although similar, using pytorch creates a large overhead in multiprocessing.
* Use float16 for the features to reduce memory usage
* Use float16 to load the dataframe and reduce memory usage
* Use sparse arrays for the features since the one-hot encodings take about 80-90% of the features

**Deprecated:**

* Previously cached datasets will no longer work.

**Removed:**

* Removed the `Predictor.training_epoch_end`. It used too much memory

**Fixed:**

* Mostly memory issues. See **Changed** section.

**Security:**

* Nothing

**Authors:**

* Dom
* Dominique



v0.2.2
====================

**Authors:**

* Hadrien Mary



v0.2.1
====================

**Authors:**

* Hadrien Mary



v0.2.0
====================

**Added:**

* Add functions and CLI to list and download datasets from Goli public GCS bucket.
* Add logic to load a pretrained model from the Goli GCS bucket.
* Add a datamodule for OGB

**Changed:**

* Save featurization args in datamodule cache and prevent reloading when the feature args are different than the one in the cache.
* Remove examples folder in doc to tutorials.

**Authors:**

* Ali
* Dom
* Dominique
* Hadrien Mary
* Hannes Stärk
* Ubuntu
* alip67



v0.1.0
====================

**Added:**

* First working version of goli. Browse the documentation and tutorials for more details.

**Authors:**

* Dom
* Hadrien Mary
* Therence1
* Ubuntu



v0.0.1
====================

**Added:**

* Fake release to test the process.

**Authors:**



