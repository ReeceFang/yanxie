# 独立的两阶段分类实验

本目录是独立实验代码，不修改也不依赖项目原有的 `config.py`、`data.py`、
`train.py`、`test.py`、`model.py`、`engine.py` 和 `utils.py`。

数据集仍保持原始目录结构：

```text
dataset/
├── train/
│   ├── 原始类别1/
│   ├── 原始类别2/
│   └── ...
└── val/
    ├── 原始类别1/
    ├── 原始类别2/
    └── ...
```

代码只在内存中修改标签，不复制、移动或改名任何图片。

## 1. 指定要合并的类别

准备一个 UTF-8 TXT，每个非空行是一个需要合并的原始类别目录名。例如：

```text
class_03
class_07
class_12
```

可以复制并修改 [merged_classes.example.txt](merged_classes.example.txt)。类别名区分大小写，
必须与 `train`、`val` 下的目录名完全一致。TXT 中至少需要两个类别，不能重复。

## 2. 训练一级分类器

假设原数据有 20 类，TXT 有 3 类，一级类别数会自动计算为
`20 - 3 + 1 = 18`，不需要传 `--num-classes`：

```bash
python hierarchical_experiment/train.py \
  --stage coarse \
  --merged-classes-file hierarchical_experiment/merged_classes.txt \
  --merged-class-name special_group \
  --data-dir /root/autodl-tmp/yanxie_data \
  --model convnext_base \
  --model-path /root/autodl-tmp/timm/convnext_base/pytorch_model.bin \
  --epochs 50 \
  --batch-size 64 \
  --learning-rate 1e-3 \
  --input-size 96 \
  --output-dir runs/hierarchical_coarse_18
```

一级训练读取全部图片，并把 TXT 中的原始类别统一映射到 `special_group`。

## 3. 训练二级分类器

```bash
python hierarchical_experiment/train.py \
  --stage fine \
  --merged-classes-file hierarchical_experiment/merged_classes.txt \
  --merged-class-name special_group \
  --data-dir /root/autodl-tmp/yanxie_data \
  --model convnext_base \
  --model-path /root/autodl-tmp/timm/convnext_base/pytorch_model.bin \
  --epochs 50 \
  --batch-size 64 \
  --learning-rate 1e-3 \
  --input-size 96 \
  --output-dir runs/hierarchical_fine_3
```

二级训练只保留 TXT 中的类别。TXT 的行顺序就是二级分类器的标签下标顺序，训练产生的
`run_config.json` 会保存该映射，推理时会自动读取。

两次训练应使用同一个 TXT 和同一个 `--merged-class-name`，但两个 `--output-dir` 必须不同。

## 4. 在完整验证集上测试方案

`--val-dir` 指向未经合并的原始 20 类验证集：

```bash
python hierarchical_experiment/cascade.py \
  --coarse-run runs/hierarchical_coarse_18 \
  --fine-run runs/hierarchical_fine_3 \
  --val-dir /root/autodl-tmp/yanxie_data/val \
  --weights best \
  --output-csv runs/hierarchical_predictions.csv
```

输出包括：

- `hierarchical_predictions.csv`：每张图的一级结果、是否进入二级、二级结果和最终结果。
- `hierarchical_predictions_summary.json`：端到端准确率、合并类别端到端准确率、一级路由
  召回率、普通类误入二级的比例、正确路由后的二级准确率和逐类准确率。

评估两阶段方案时，应把 `overall_accuracy` 与原始 20 分类器在同一个验证集上的准确率比较。
同时查看：

- `merged_route_recall`：真实属于合并三类的图片有多少成功进入二级。
- `fine_accuracy_given_correct_route`：成功进入二级以后，三分类器本身的效果。
- `other_route_false_positive_rate`：普通 17 类被错误送入二级的比例。
- `merged_subset_end_to_end_accuracy`：把一级漏检也计算在内的合并三类最终准确率。

## 5. 对无标签图片推理

`--input` 可以接一张或多张图片，也可以接目录；目录会递归搜索：

```bash
python hierarchical_experiment/cascade.py \
  --coarse-run runs/hierarchical_coarse_18 \
  --fine-run runs/hierarchical_fine_3 \
  --input /path/to/image.jpg /path/to/another_directory \
  --weights best \
  --output-csv runs/inference_predictions.csv
```

只有一级 Top-1 为 `special_group` 的图片才会执行二级模型。命中二级时，CSV 中的
`final_confidence` 为一级合并类概率乘以二级条件概率。

## 实验注意事项

三个原始类合并后，一级模型中的合并类样本量可能明显大于普通类，容易产生类别偏置。
训练日志会输出一级训练集每个标签的样本数。第一轮验证方案时建议先保持与原始 20 分类
基线相同的数据划分、模型、预训练权重、输入尺寸和训练超参数，只改变标签结构，这样对比
才有意义。
