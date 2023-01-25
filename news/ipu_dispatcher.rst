**Added:**

* support new compiler based on dispatcher
* support for latest pytorch-lightning using strategies instead of plugins
* Simplified IPU installation. Specific IPU requirements file
* IPU tutorial script along side notebook

**Removed:**

* older IPU compiler based on tracer
* need to convert custom data structure to tuple of Tensors in order to use the IPU
