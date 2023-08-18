# Graphium pretrained models

Graphium aims to provide a set of pretrained models that you can use for inference or transfer learning. The models will be made available once trained and validated.

## Listing all available models

To know which models are available, you can run the following command

```python
import graphium

print(graphium.trainer.PredictorModule.list_pretrained_models())
```

## Dummy pre-trained model

At the moment, only `tests/dummy-pretrained-model.ckpt` is provided, which is mostly useful for development and debugging of the checkpointing and finetuning pipelines.

You can load a pretrained models using the Graphium API:

```python
import graphium

predictor = graphium.trainer.PredictorModule.load_pretrained_models("dummy-pretrained-model")
```
