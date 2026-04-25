### 配置环境
```
conda env create -f environment_linux.yaml
```

### 激活环境
```
conda activate SSTNet
```

### 开始训练
指定单个yaml训练：
```
python -m cli.train --config yaml/<config_name>.yaml
```

指定某个 YAML 目录批量训练：
```
python -m cli.train --config-dir <yaml_file_name>
```
会自动读取目录下所有的yaml文件并运行。同时权重文件和log文件会按照对应读取的yaml文件名自动命名区分，过程中无需手动操作。

权重文件位置: `.checkpoints_<yaml文件名>/`

log日志文件位置: `.logs/<yaml文件名>`

若意外中断，可以将已跑好的yaml文件移除 yaml_file_name/ 文件夹后继续训练



训练共100epoch，早停patience=10，也就是早停后第 n-10 个文件为 best_checkpoint

> yaml文件命名规则：序号+数据集代号+模型名+内插或外插.yaml，S指SST数据集，T指TurbulentFlow数据集