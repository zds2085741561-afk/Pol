# Pol: Sequential POI Recommendation

序列兴趣点推荐实验代码，基于 [EffiPOI](https://github.com/pcm1217/EffiPOI) 扩展，包含教师—学生知识蒸馏、乘积量化、轨迹流增强和多兴趣建模。

## 目录

教师默认启用轨迹流图、多兴趣建模和局部重排序。学生支持不同蒸馏与量化设置，运行参数见各入口的 `--help`。

| 路径 | 内容 |
| --- | --- |
| `teacher.py`、`teacher_main.py` | 教师模型、训练与评估 |
| `student.py`、`student_main.py`、`trainer.py` | 学生模型、知识蒸馏与训练 |
| `pq.py`、`data/` | 量化模块和 RecBole 数据适配 |
| `props/` | 教师、学生及通用配置 |
| `tools/` | STEPS 数据转换和下载辅助工具 |
| `baselines/` | CL-SASRec、MIMAR、S2HyRec、BSARec 实验适配代码 |
| `continuous_base/` | 基础学生的连续推荐评估与调参 |
| `continuous_distilled/` | 蒸馏学生的连续推荐评估与调参 |
| `experiments/` | 筛选后的历史日志、结果文件及来源校验索引 |

## 环境准备

先安装适合本机 CPU/CUDA 的 PyTorch，再安装其余依赖：

```bash
git clone https://github.com/zds2085741561-afk/Pol.git
cd Pol
python -m pip install -r requirements.txt
```

`requirements.txt` 是依赖清单，尚不是经过全新环境验证的版本锁。发布时本机可导入的版本为 PyTorch `2.7.0.dev20250310+cu124`、RecBole `1.2.0`、Faiss `1.8.0`；这不代表所有历史运行均使用该环境。

## 数据准备

本仓库不包含原始签到记录、特征矩阵和模型权重。请准备具有使用权限的数据，保持训练、验证和测试划分一致。入口默认数据集名称为 `NYC`，也支持项目中的 `NYC_STEPS` 配置。所需文件放置示例：

```text
data/NYC/
  NYC.train.inter
  NYC.valid.inter
  NYC.test.inter
  NYC.feat1CLS
  NYC.feat_token_to_row.json
  NYC.OPQ128,IVF1,PQ128x4.strict.index
```

交互字段配置为 `user_id`、`item_id_list`、`item_id`，详见 `props/fintune.yaml`。特征为 1664 维 float32，token 映射用于对齐特征行和 RecBole 内部物品编号。数据根目录已改为相对路径 `./data/`，请从仓库根目录运行命令。

STEPS 转换入口：

```bash
python tools/steps_effi_builder_v2/build_steps_effi_files_v2.py --help
```

该工具生成确定性哈希文本特征，并非预训练语言模型特征。正式实验需准备完整特征和索引；数据加载器的随机特征或零编码回退不能用于有效复现。

## 训练与评估

训练完整教师（从头训练时显式启用编码器训练）：

```bash
python teacher_main.py -d NYC --full_train --force_cuda --epochs 300
```

使用实际生成的教师检查点训练学生；以下检查点名称是占位符，需替换：

```bash
python student_main.py -d NYC -p saved/TEACHER_CHECKPOINT.pth --force_cuda --epochs 300
```

教师评估与学生测试：

```bash
python teacher_main.py -d NYC -p saved/TEACHER_CHECKPOINT.pth --eval_only --force_cuda
python student_main.py -d NYC -p saved/TEACHER_CHECKPOINT.pth --resume_student saved/STUDENT_CHECKPOINT.pth --test_only --force_cuda
```

CPU 可使用 `--cpu` 替代 `--force_cuda`。全部选项请运行 `python teacher_main.py --help` 或 `python student_main.py --help`。通用评估配置报告 HIT 和 NDCG，截断位置为 1、5、10、20，默认验证指标为 HIT@10；具体历史实验以日志中的有效配置为准。

调参应使用验证集，学生入口提供 `--valid_only`；确定配置后再进行测试集评估。这里提供运行入口，不承诺当前默认配置自动复现全部历史结果。

## 实验记录

`experiments/` 包含 43 份完成测试的历史日志和 11 份结果 JSON。日志保留参数和指标，未按成绩高低筛选；校验信息见 `experiments/log_manifest.json`。

历史日志包含探索性实验，尚未逐一对应论文表格；复现时请核对代码版本、数据和参数。

## 连续推荐

两套实现分别保留基础学生与蒸馏学生的加载设置。指定自己的检查点进行评估或验证集调参：

```bash
python continuous_base/evaluate.py --help
python continuous_base/tune.py --help
python continuous_distilled/evaluate.py --help
python continuous_distilled/tune.py --help
```

## 验证与来源

已检查 Python 语法和命令行入口，尚未完成全量重新训练及跨机器复现。

基础实现来源：[EffiPOI](https://github.com/pcm1217/EffiPOI)。本仓库的扩展代码与对比实验适配不应被表述为全部从零原创。仓库公开不等于授予任意再分发许可；在补充许可证前，请核对基础项目和各依赖的使用条款。

代码公开地址：<https://github.com/zds2085741561-afk/Pol>
