# INCREASING BOTH BATCH SIZE AND LEARNING RATE ACCELERATES STOCHASTIC GRADIENT DESCENT
Source code for reproducing our paper's experiments.
# Abstract
The performance of mini-batch stochastic gradient descent (SGD) strongly depends on the setting of batch size and learning rate to minimize the empirical loss in training a deep neural network. In this paper, we give theoretical analyses of mini-batch SGD with four schedulers, (i) constant batch size and decaying learning rate scheduler, (ii) increasing batch size and decaying learning rate scheduler, (iii) increasing batch size and increasing learning rate scheduler, and (iv) increasing batch size and warmup decaying learning rate scheduler. We show that mini-batch SGD using scheduler (i) does not always minimize the expectation of the full gradient norm of the empirical loss, while mini-batch SGD using each of schedulers (ii), (iii), and (iv) minimizes it. In particular, using scheduler (iii) and (iv) accelerates mini-batch SGD. We provide numerical results supporting the analyses such that using schedulers (iii) and (iv) minimizes the full gradient norm of the empirical loss faster than using other schedulers.
# Usage
To train a model using the CIFAR-100 dataset, run the `cifar100.py` script with a specified JSON file that contains the training parameters:
```
python cifar100.py XXXXX.json
```
To resume training from a checkpoint, add the `--resume` option to the command. This will load the model state from the checkpoint specified in `checkpoint_path` and continue training from that point:
```
python cifar100.py XXXXX.json --resume
```
For more details about the checkpoint, refer to the `checkpoint_path` section in the **Parameters Description**.

To customize the training process, modify the parameters in the JSON file and rerun the script. You can adjust the model architecture, learning rate, batch size, and other parameters to explore different training strategies and observe their effects on model performance.
## Example JSON Configuration
The following JSON configuration file is located at `json/incr_bs_warmup_lr/warmup_const_lr_max0.2.json`:
```
{
    "model": "WideResNet28_10",
    "case": "incr_bs_warmup_lr",
    "lr": 0.1,
    "lr_max": 0.2,
    "epochs": 300,
    "incr_interval": 30,
    "warmup_epochs": 30,
    "warmup_interval": 3,
    "batch_size": 8,
    "bs_max": 1024,
    "checkpoint_path": "checkpoint/warmup_const_lr_max0.2.pth.tar",
    "lr_method": "warmup_const",
    "bs_delta": 2.0
}
```
### Parameters Description
| Parameter | Value | Description |
| :-------- | :---- | :---------- |
| `model` | `"WideResNet28_10"` or `"ResNet18"` |Specifies the model architecture to use. Options are `WideResNet28_10` or `ResNet18`.|
| `case` |`"const_bs_decay_lr"`,<br>`"incr_bs_decay_lr"`,<br>`"incr_bs_incr_lr"`,<br>`"incr_bs_warmup_lr"`|Describes the training strategy:<br> - `"const_bs_decay_lr"`: Constant batch size with decaying learning rate.<br> - `"incr_bs_decay_lr"`: Increasing batch size with decaying learning rate.<br> - `"incr_bs_incr_lr"`: Increasing both batch size and learning rate.<br> - `"incr_bs_warmup_lr"`: Increasing batch size with a learning rate warmup.|
|`lr`|`float` (e.g., `0.1`)|The initial learning rate for the optimizer.|
|`lr_max`|`float` (e.g., `0.2`)|The maximum learning rate to be reached when the learning rate is increasing. Used when `case` is `"incr_bs_incr_lr"` or `"incr_bs_warmup_lr"`.|
|`epochs`|`int` (e.g., `300`)|The total number of epochs for training.|
|`incr_interval`|`int` (e.g., `30`)|Interval (in epochs) at which the batch size will increase. Also, the interval for increasing the learning rate when `case` is `"incr_bs_incr_lr"`. Used when `case` is `"incr_bs_decay_lr"`, `"incr_bs_incr_lr"`, or `"incr_bs_warmup_lr"`.|
|`warmup_epochs`|`int` (e.g., `30`)|Number of epochs over which the learning rate warms up from `lr` to `lr_max`. Used when `case` is `"incr_bs_warmup_lr"`.|
|`warmup_interval`|`int` (e.g., `3`)|The interval (in epochs) during which the learning rate increases in the warmup phase. Used when `case` is `"incr_bs_warmup_lr"`.|
|`batch_size`|`int` (e.g., `8`)|The initial batch size at the beginning of training.|
|`bs_max`|`int` (e.g., `1024`)|The maximum batch size allowed during training. Used when `case` is `"incr_bs_decay_lr"`, `"incr_bs_incr_lr"`, or `"incr_bs_warmup_lr"`.|
|`checkpoint_path`|`str` (e.g., `"checkpoint/XXXXX.pth.tar"`)|Specifies any `"pth.tar"` file in the `checkpoint` directory. Checkpoints are saved at each epoch. If `--resume` is added to the command (`python cifar100.py json/XXXXX.json --resume`), training can be resumed from the checkpoint.|
|`lr_method`|`"constant"`, `"cosine"`, `"diminishing"`, `"linear"`, `"poly"`, <br>`"exp_growth"`, `"triply_incr_bs"`, `"quadruply_incr_bs"`,<br>`"warmup_const"`, `"warmup_cosine"`|Method for adjusting the learning rate. The options depend on the `case`:<br><br> - `"const_bs_decay_lr"` or `"incr_bs_decay_lr"`:<br> `"constant"`, `"cosine"`, `"diminishing"`, `"linear"`, `"poly"`.<br><br> - `"incr_bs_incr_lr"`:<br> `"exp_growth"`, `"triply_incr_bs"`, `"quadruply_incr_bs"`.<br><br> - `"incr_bs_warmup_lr"`:<br> `"warmup_const"`, `"warmup_cosine"`.|
|`bs_delta`|`float` (e.g., `2.0`)|The factor by which the batch size increases after each interval. Used when `case` is `"incr_bs_decay_lr"`, `"incr_bs_incr_lr"`, or `"incr_bs_warmup_lr"`.|
|`power`| `float` (e.g., `2.0`) |A parameter used when `lr_method` is set to `"poly"`, defining the polynomial decay rate of the learning rate.|
