# MSGNet

Implementation of the Multiscale Guidance Network (MSG-Net) from the ["Depth Map Super-Resolution by Deep Multi-Scale Guidance"](https://github.com/twhui/MSG-Net) project.

> Super-Resolution ONLY! No Inpainting, data is assumed to be hole-filled already.

```
msgnet
├── saved_models
├── utils
│   ├── __init__.py
│   ├── arg_parser.py
│   ├── dataset_loader.py
|   ├── functional.py
|   ├── losses.py
│   └── utils.py
├── networks
│   ├── __init__.py
│   ├── layers.py
│   └── MSGNet.py
├── model_test.ipynb
├── generate_custom_set.ipynb
├── training_procedure.py
├── train_MSGNet.py
└── README.md
```
> The `saved_models` folder is empty by default, all models must be saved or moved there.
