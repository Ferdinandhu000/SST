### 配置环境
```
conda env create -f environment_linux.yaml
```

### 激活环境
```
conda activate SSTNet
```

### 开始训练
依次选用8个yaml文件，例如第一个yaml文件 `1-FNO_inside_config.yaml`
```
python -m cli.train --config=1-FNO_inside_config.yaml
```
在`config=`后更换yaml配置文件名即可

注意训练完后一定要更改.checkpoints/目录名称，否则进行下一次训练后权重文件会覆盖源文件中的内容。例如第一个配置文件训练完成后，可以参考配置文件将目录命名为.checkpoints_1-FNO_inside/ 

训练共100epoch，早停patience=10，也就是早停后第 n-10 个文件为 best_checkpoint