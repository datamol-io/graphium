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
