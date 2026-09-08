# Pol: Sequential POI Recommendation

面向序列兴趣点推荐的研究代码库，包含教师—学生知识蒸馏、乘积量化（PQ）、轨迹流图增强、多兴趣建模及重排序实验。本仓库基于 [EffiPOI](https://github.com/pcm1217/EffiPOI) 进行扩展，用于代码公开、实验检查和后续复现。

## 发布内容

当前发布的是本地研究代码快照。教师默认配置启用 GTF、MIC 和 MIFG；学生入口支持完整学生模型以及 TF-KD、TF-PQ、TFCA、CMID 等实验开关。不同实验需使用对应参数，不能仅凭文件名认定为论文中的某一行结果。

| 路径 | 内容 |
| --- | --- |
| `teacher.py`、`teacher_main.py` | 教师模型、训练与评估 |
| `student.py`、`student_main.py`、`trainer.py` | 学生模型、知识蒸馏与训练 |
| `pq.py`、`data/` | 量化模块和 RecBole 数据适配 |
| `props/` | 教师、学生及通用配置 |
| `tools/` | STEPS 数据转换和下载辅助工具 |
| `baselines/` | CL-SASRec、MIMAR、S2HyRec、BSARec 实验适配代码 |
| `innovation2_e3/`、`innovation2_s3_kf/` | 连续推荐及验证集调参脚本 |
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

该工具生成的是确定性哈希文本特征，不能将其描述为预训练语言模型生成的语义特征。原有数据适配器在缺失特征或索引时有随机特征/零编码回退；正式实验必须先检查输入文件与运行日志，回退运行不能作为有效复现。根目录 `build_index.py` 是历史辅助脚本，含固定检查点和输出命名，不是通用的一键数据准备入口。

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

本次整理保留 43 份包含最终测试结果、无 traceback/OOM 且内容不重复的历史日志，以及 11 份最终/测试/复核结果 JSON。筛选未按测试成绩高低进行，部分记录可能是探索性实验。日志保留原始参数与指标，用户目录信息进行脱敏；`experiments/log_manifest.json` 记录原文件和发布文件的 SHA-256。

历史日志来自多个时期，尚未逐一建立“代码版本—参数—论文表格”对应关系，因此本 README 不将其汇总为已验证的论文结果。空日志、仅初始化日志、无最终测试结果的运行、模型权重及 TensorBoard 缓存未纳入本次发布；本地原件保留。

## 验证状态与来源

发布检查包含 Python 语法解析、命令行帮助检查和文件筛选检查，未重新训练所有模型，也未完成跨机器端到端复现。依赖、数据特征和历史参数差异可能影响结果。

基础实现来源：[EffiPOI](https://github.com/pcm1217/EffiPOI)。本仓库的扩展代码与对比实验适配不应被表述为全部从零原创。仓库公开不等于授予任意再分发许可；在补充许可证前，请核对基础项目和各依赖的使用条款。

代码公开地址：<https://github.com/zds2085741561-afk/Pol>
