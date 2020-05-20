# Human Segmentation in Pytorch

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [Human Segmentation in Pytorch](#pytorch-template-project)
	* [Requirements](#requirements)
	* [Features](#features)
	* [Folder Structure](#folder-structure)
	* [Usage](#usage)
		* [Config file format](#config-file-format)
		* [Using config files](#using-config-files)
		* [Resuming from checkpoints](#resuming-from-checkpoints)
    * [Using Multiple GPU](#using-multiple-gpu)
	* [Customization](#customization)
		* [Custom CLI options](#custom-cli-options)
		* [Data Loader](#data-loader)
		* [Trainer](#trainer)
		* [Model](#model)
		* [Loss](#loss)
		* [metrics](#metrics)
		* [Additional logging](#additional-logging)
		* [Validation data](#validation-data)
		* [Checkpoints](#checkpoints)
    * [Tensorboard Visualization](#tensorboard-visualization)
	* [Contribution](#contribution)
	* [TODOs](#todos)
	* [License](#license)
	* [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->

## Requirements
* Python >= 3.5 (3.6 recommended)
* PyTorch >= 0.4 (1.2 recommended)
* tqdm
* tensorboard >= 1.14 (see [Tensorboard Visualization](#tensorboard-visualization))

## Features
* A clear and easy to navigate structure.
* A `.json` config file with a lot of possibilities for parameter tuning.
* Supports various models, losses, Lr schedulers, data augmentations.
* Checkpoint saving and resuming.
* Abstract base classes for faster development:
  * `BaseTrainer` handles checkpoint saving/resuming, training process logging, and more.
  * `BaseDataLoader` handles batch generation, data shuffling, and validation data splitting.
  * `BaseModel` provides basic model summary.

## Folder Structure
  ```
  humanseg.pytorch/
  │
  ├── base
  │   ├── base_data_loader.py
  │   ├── base_inference.py
  │   ├── base_model.py
  │   ├── base_trainer.py
  │   └── __init__.py
  ├── config
  │
  ├── data_loader
  │   ├── data_loaders.py
  │   └── transforms.py
  ├── LICENSE
  ├── logger
  │   ├── __init__.py
  │   ├── logger_config.json
  │   ├── logger.py
  │   └── visualization.py
  ├── README.md
  ├── test.py
  ├── trainer
  │   ├── __init__.py
  │   └── trainer.py
  ├── train.py
  └── utils
      ├── __init__.py
      ├── loss.py
      ├── metric.py
      └── util.py
  ```

## Usage
The code in this repo is an MNIST example of the template.
Try `python train.py -c config.json` to run code.

### Config file format
Config files are in `.json` format:
```javascript
{
    "name": "HumanSeg",
    "n_gpu": 1,
    
    "arch": {
        "type": "UNet",
        "args": {
            "backbone": "mobilenetv2",
            "num_classes": 2,
            "pretrained_backbone": null
        }
    },

	"train_loader": {
		"type": "SegmentationDataLoader",
		"args":{
			"pairs_file": "dataset/train_mask.txt",
			"color_channel": "RGB",
			"resize": 320,
			"padding_value": 0,
			"is_training": true,
			"noise_std": 3,
			"crop_range": [0.90, 1.0],
			"flip_hor": 0.5,
			"rotate": 0.0,
			"angle": 10,
			"normalize": true,
			"one_hot": false,
			"shuffle": true,
			"batch_size": 16,
			"n_workers": 24,
			"pin_memory": true
		}
	},

	"valid_loader": {
		"type": "SegmentationDataLoader",
		"args":{
			"pairs_file": "dataset/valid_mask.txt",
			"color_channel": "RGB",
			"resize": 320,
			"padding_value": 0,
			"is_training": false,
			"normalize": true,
			"one_hot": false,
			"shuffle": false,
			"batch_size": 16,
			"n_workers": 24,
			"pin_memory": true
		}
	},

	"optimizer": {
		"type": "SGD",
		"args":{
			"lr": 1e-2,
			"momentum": 0.9,
			"weight_decay": 1e-8
		}
	},

	"loss": "dice_loss",
	"metrics": [
		"miou"
	],

	"lr_scheduler": {
		"type":"StepLR",
		"args":{
			"step_size": 100,
			"gamma": 1.0
		}
	},

	"trainer": {
		"epochs": 80,
		"save_dir": "/workspace/checkpoints/",
		"save_freq": null,
		"verbosity": 2,
		"monitor": "valid_loss",
		"monitor_mode": "min"
	},

	"visualization":{
		"tensorboardX": true,
		"log_dir": "/workspace/checkpoints/"
	}
}
```

Add addional configurations if you need.

### Using config files
Modify the configurations in `.json` config files, then run:

  ```
  python train.py --config config.json
  ```

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```
  python train.py --resume path/to/checkpoint
  ```

### Using Multiple GPU
You can enable multi-GPU training by setting `n_gpu` argument of the config file to larger number.
If configured to use smaller number of gpu than available, first n devices will be used by default.
Specify indices of available GPUs by cuda environmental variable.
  ```
  python train.py --device 2,3 -c config.json
  ```
  This is equivalent to
  ```
  CUDA_VISIBLE_DEVICES=2,3 python train.py -c config.py
  ```

## Customization

### Custom CLI options

Changing values of config file is a clean, safe and easy way of tuning hyperparameters. However, sometimes
it is better to have command line options if some values need to be changed too often or quickly.

This template uses the configurations stored in the json file by default, but by registering custom options as follows
you can change some of them using CLI flags.

  ```python
  # simple class-like object having 3 attributes, `flags`, `type`, `target`.
  CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
  options = [
      CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
      CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
      # options added here can be modified by command line flags.
  ]
  ```
`target` argument should be sequence of keys, which are used to access that option in the config dict. In this example, `target` 
for the learning rate option is `('optimizer', 'args', 'lr')` because `config['optimizer']['args']['lr']` points to the learning rate.
`python train.py -c config.json --bs 256` runs training with options given in `config.json` except for the `batch size`
which is increased to 256 by command line options.


### Data Loader
* **Writing your own data loader**

1. **Inherit ```BaseDataLoader```**

    `BaseDataLoader` is a subclass of `torch.utils.data.DataLoader`, you can use either of them.

    `BaseDataLoader` handles:
    * Generating next batch
    * Data shuffling
    * Generating validation data loader by calling
    `BaseDataLoader.split_validation()`

* **DataLoader Usage**

  `BaseDataLoader` is an iterator, to iterate through batches:
  ```python
  for batch_idx, (x_batch, y_batch) in data_loader:
      pass
  ```
* **Example**

  Please refer to `data_loader/data_loaders.py` for an MNIST data loading example.

### Trainer
* **Writing your own trainer**

1. **Inherit ```BaseTrainer```**

    `BaseTrainer` handles:
    * Training process logging
    * Checkpoint saving
    * Checkpoint resuming
    * Reconfigurable performance monitoring for saving current best model, and early stop training.
      * If config `monitor` is set to `max val_accuracy`, which means then the trainer will save a checkpoint `model_best.pth` when `validation accuracy` of epoch replaces current `maximum`.
      * If config `early_stop` is set, training will be automatically terminated when model performance does not improve for given number of epochs. This feature can be turned off by passing 0 to the `early_stop` option, or just deleting the line of config.

2. **Implementing abstract methods**

    You need to implement `_train_epoch()` for your training process, if you need validation then you can implement `_valid_epoch()` as in `trainer/trainer.py`

* **Example**

  Please refer to `trainer/trainer.py` for MNIST training.

* **Iteration-based training**

  `Trainer.__init__` takes an optional argument, `len_epoch` which controls number of batches(steps) in each epoch.

### Model
* **Writing your own model**

1. **Inherit `BaseModel`**

    `BaseModel` handles:
    * Inherited from `torch.nn.Module`
    * `__str__`: Modify native `print` function to prints the number of trainable parameters.

2. **Implementing abstract methods**

    Implement the foward pass method `forward()`

* **Example**

  Please refer to `model/model.py` for a LeNet example.

### Loss
Custom loss functions can be implemented in 'model/loss.py'. Use them by changing the name given in "loss" in config file, to corresponding name.

### Metrics
Metric functions are located in 'model/metric.py'.

You can monitor multiple metrics by providing a list in the configuration file, e.g.:
  ```json
  "metrics": ["accuracy", "top_k_acc"],
  ```

### Additional logging
If you have additional information to be logged, in `_train_epoch()` of your trainer class, merge them with `log` as shown below before returning:

  ```python
  additional_log = {"gradient_norm": g, "sensitivity": s}
  log.update(additional_log)
  return log
  ```

### Testing
You can test trained model by running `test.py` passing path to the trained checkpoint by `--resume` argument.

### Validation data
To split validation data from a data loader, call `BaseDataLoader.split_validation()`, then it will return a data loader for validation of size specified in your config file.
The `validation_split` can be a ratio of validation set per total data(0.0 <= float < 1.0), or the number of samples (0 <= int < `n_total_samples`).

**Note**: the `split_validation()` method will modify the original data loader
**Note**: `split_validation()` will return `None` if `"validation_split"` is set to `0`

### Checkpoints
You can specify the name of the training session in config files:
  ```json
  "name": "MNIST_LeNet",
  ```

The checkpoints will be saved in `save_dir/name/timestamp/checkpoint_epoch_n`, with timestamp in mmdd_HHMMSS format.

A copy of config file will be saved in the same folder.

**Note**: checkpoints contain:
  ```python
  {
    'arch': arch,
    'epoch': epoch,
    'state_dict': self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'monitor_best': self.mnt_best,
    'config': self.config
  }
  ```

### Tensorboard Visualization
This template supports Tensorboard visualization by using either  `torch.utils.tensorboard` or [TensorboardX](https://github.com/lanpa/tensorboardX).

1. **Install**

    If you are using pytorch 1.1 or higher, install tensorboard by 'pip install tensorboard>=1.14.0'.

    Otherwise, you should install tensorboardx. Follow installation guide in [TensorboardX](https://github.com/lanpa/tensorboardX).

2. **Run training** 

    Make sure that `tensorboard` option in the config file is turned on.

    ```
     "tensorboard" : true
    ```

3. **Open Tensorboard server** 

    Type `tensorboard --logdir saved/log/` at the project root, then server will open at `http://localhost:6006`

By default, values of loss and metrics specified in config file, input images, and histogram of model parameters will be logged.
If you need more visualizations, use `add_scalar('tag', data)`, `add_image('tag', image)`, etc in the `trainer._train_epoch` method.
`add_something()` methods in this template are basically wrappers for those of `tensorboardX.SummaryWriter` and `torch.utils.tensorboard.SummaryWriter` modules. 

**Note**: You don't have to specify current steps, since `WriterTensorboard` class defined at `logger/visualization.py` will track current steps.

## Contribution
Feel free to contribute any kind of function or enhancement, here the coding style follows PEP8

Code should pass the [Flake8](http://flake8.pycqa.org/en/latest/) check before committing.

## TODOs

- [ ] Multiple optimizers
- [ ] Support more tensorboard functions
- [x] Using fixed random seed
- [x] Support pytorch native tensorboard
- [x] `tensorboardX` logger support
- [x] Configurable logging layout, checkpoint naming
- [x] Iteration-based training (instead of epoch-based)
- [x] Adding command line option for fine-tuning

## License
This project is licensed under the MIT License. See  LICENSE for more details

## Acknowledgements
This project is inspired by the project [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) by [Mahmoud Gemy](https://github.com/MrGemy95)
