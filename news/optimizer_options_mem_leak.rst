**Added:**

* Support for OptimOptions

**Fixed:**

* `OptimOptions` fixed, with a better handling of the different options
* Fixed a memory leak during the dataloading due to using `List[str]` in the `Dataset`
* Removed the `sample_weights` from the `average_precision_ipu` since it was unused and depreciated in `torchmetrics=0.10`
