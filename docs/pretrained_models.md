# GOLI pretrained models

GOLI provides a set of pretrained models that you can use for inference or transfer learning. The models are available on Google Cloud Storage at `gs://goli-public/pretrained-models`.

You can load a pretrained models using the GOLI API:

```python
import goli

predictor = goli.trainer.PredictorModule.load_pretrained_models("goli-zinc-micro-dummy-test")
```

## `goli-zinc-micro-dummy-test`

Dummy model used for testing purposes _(probably to delete in the future)_.
