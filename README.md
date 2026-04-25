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

### 打包权重文件
训练完成后得到大量.checkpoints_<yaml_name>目录，执行以下命令可以快速提取每个目录中的最佳权重文件（即倒数第11个文件）并打包至 checkpoints_best/ 目录下，方便汇总。

log日志文件全部保存在 .logs/ 目录下，一并复制到checkpoints_best/ 中即可
```
python pack.py <checkpoints目录文件名1> <checkpoints目录文件名2> ...
# 例如想要打包 `.checkpoints_10-T-CATO_fno_outside_config`,
# `.checkpoints_11-T-CATO_afno_inside_config`,
# '.checkpoints_12-T-CATO_afno_outside_config'
# 这三个目录下的最佳权重文件，仅需运行
# python pack.py .checkpoints_10-T-CATO_fno_outside_config .checkpoints_11-T-CATO_afno_inside_config .checkpoints_12-T-CATO_afno_outside_config
# 注意是空格隔开
```
返回得到 checkpoints_best/ 目录，结构如下
```
checkpoints_best/
└── .checkpoints_10-T-CATO_fno_outside_config/
    └── xxx.pt          ← 仅一个最佳权重文件
    .checkpoints_11-T-CATO_afno_inside_config
    └── yyy.pt 
    .checkpoints_12-T-CATO_afno_outside_config
    └── zzz.pt 
```
也就是只需要在`python pack.py`后加上需要打包的checkpoints目录名并用空格隔开即可