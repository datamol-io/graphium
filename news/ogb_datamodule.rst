**Added:**

* Added file ipu_simple_lightning.py to test IPU / lightning implementations
* Added gradient accumulation from the config file for IPUs
* Added class `IPUDataModuleModifier` to handle IPU's using dual inheritance
* Re-added the unit-test for `GraphOGBDataModule`

**Changed:**

* Renamed `batch_size_train_val` and `batch_size_test` to `batch_size_inference` and `batch_size_training` for consistency with IPU options
* Refactored `MultitaskFromSmilesDataModule` to handle IPU's using dual inheritance

**Removed:**

* Removed TaskHead class, which was redundant with the FeedForwardNN class

**Fixed:**

* Fixed PNA implementation
* Fixed logging of the training loss using `on_train_batch_end` instead of `training_step`
* Fixed the class `GraphOGBDataModule` to handle CPU, GPU, and IPU's, and support for multi-task (only multi-ogb tasks for now)
