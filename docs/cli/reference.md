# CLI Reference

Installing the Graphium library, makes two CLI tools available. 

- [`graphium-train`](./graphium-train.md) is the hydra endpoint - specifically meant for training, finetuning and testing. Since this uses `@hydra.main`, it has access to all advanced hydra functionality such as tab completion, multirun, working directory management, logging management. 
- [`graphium`](./graphium.md) is the more general CLI endpoint, organized with various sub commands. 

Ideally, we would've integrated both in a single CLI endpoint, but the hydra CLI cannot be a subcommand of another CLI, nor does it support easily adding subcommands, which is why provide two separate CLI tools with different purposes.

!!! note "Interactive, embedded CLI docs with `--help`"
    In addition to these pages, you can also use `graphium --help` and `graphium-train --help` to interactively navigate the documentation of these tools directly in the CLI.