### 配置环境
```
conda env create -f environment_linux.yaml
```

### 激活环境
```
conda activate SSTNet
```

### 开始训练
11-20 的yaml文件已存入yaml/ 目录中，终端运行
```
python -m cli.train
```
会自动读取目录下所有的yaml文件并运行。同时权重文件会自动命名区分，过程中无需手动操作。

若意外终端，要将已跑好的yaml文件移除yaml/ 文件夹


训练共100epoch，早停patience=10，也就是早停后第 n-10 个文件为 best_checkpoint