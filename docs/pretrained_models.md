# Graphium pretrained models

Graphium provides a set of pretrained models that you can use for inference or transfer learning. The models are available on Google Cloud Storage at `gs://graphium-public/pretrained-models`.

You can load a pretrained models using the Graphium API:

```python
import graphium

predictor = graphium.trainer.PredictorModule.load_pretrained_models("graphium-zinc-micro-dummy-test")
```

## `graphium-zinc-micro-dummy-test`

Dummy model used for testing purposes _(probably to delete in the future)_.
