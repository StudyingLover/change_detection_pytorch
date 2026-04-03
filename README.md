# Change Detection PyTorch

fork from [likyoo/change_detection.pytorch](https://github.com/likyoo/change_detection.pytorch)

## Start train

The detailed training instructions can be found in `train_app.py`. Here is an example command to start training on the LEVIR-CD dataset:

```bash
python train_app.py --dataset LEVIR-CD --data_dir /path/to/LEVIR-CD+ --model model_name
```

## Custom model

You can view `change_detection_pytorch/applications/casp/model.py`.

All in one, use the `@register_model('your_model_name')` decorator to register your model, and then you can specify `--model your_model_name` when training.
