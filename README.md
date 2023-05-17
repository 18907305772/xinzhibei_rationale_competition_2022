# “兴智杯”全国人工智能创新应用大赛 -- 深度学习模型可解释性

团队：中山大学语言智能团队

最终排名：第三名（二等奖）

答辩 PPT：[PPT](深度学习模型可解释性赛-中山大学语言智能团队-万凡琦.pptx)

知乎 Link：[Link](https://zhuanlan.zhihu.com/p/630148578)

这个仓库是我们团队在 [“兴智杯”全国人工智能创新应用大赛 -- 深度学习模型可解释性](http://www.aiinnovation.com.cn/#/trackDetail?id=23) 比赛中取得第三名结果的相关文件及代码开源，包含我们完成本次比赛的完整思路和复现代码。

## 整体思路

### 引言

深度学习技术大幅提升了计算机视觉、自然语言处理、智能语音等应用任务的感知能力，技术成果已成功应用于多个行业场景。然而，由于深度学习模型内部预测机制不透明，导致基础核心算法在行业落地中的准确度、鲁棒性、可靠性难以进一步突破，同时也由于模型决策原因难以追溯，有关应用面临信任危机。因此，加强算法可解释性分析是加快人工智能规模化应用的重要推进路径。本赛题为自然语言处理领域的中文相似度计算模型可解释性赛题，旨在判断两条输入文本相似程度的同时给出做出答案判断的依据文本片段。针对这句子相似度问题，本团队首先使用互联网大规模中文相似度数据预训练，然后使用lcqmc数据集在句子相似度任务上微调，同时使用技术手段提高模型的鲁棒性。针对可解释性问题，本团队使用基于梯度的归因方法，给出高置信度的文本作为依据。

### 方法

#### 数据

本团队在预训练阶段使用了互联网上的大规模中文相似度数据，具体如下：DIAC2019、CCKS2018_Task3、LCQMC、AFQMC、GAIIC2021_Task3、THS2021、CHIP2019、SOHU_2021、COVID19、PAWSX、XF2021、Chinese-MNLI、Chinese-SNLI、Chinese-STS-B、OCNLI、PKU Paraphrase Bank。针对4个NLI数据集进行标签转换，STS-B数据集进行分数过滤，PKU复述数据集仅生成正标签，数据地址位于：[Link](https://github.com/liucongg/NLPDataSet)。在微调阶段，我们使用lcqmc数据集训练，同时使用传递增强策略，对训练集做数据增强，具体而言，如果句子1和句子2是相似样本，句子2和句子3是相似样本，则增强样例为句子1和句子3是相似样本；如果句子1和句子2是相似样本，句子2和句子3是不相似样本，则句子1和句子3是不相似样本。同时为了保证数据分布一致性，我们对增强结果按照原始数据集标签样本比例进行采样得到最后的增强数据集。

#### 模型

系统框架主要分为三个部分，分别是继续预训练，句子对相似度计算和归因打分。

继续预训练部分我们使用大规模中文相似度数据使用和RoBERTa相同的MLM方式对chinese_roberta_wwm_ext_large继续进行领域数据预训练，从而达到消除预训练和下游任务之间不一致的目的。训练超参数如下：训练轮数为60，学习率为5e-5，batch size为64，梯度累计步数为4，最大句子长度为64。

句子相似度计算部分我们使用pair-wise的交互式编码方式，具体而言，我们首先将两个句子拼接，然后使用[CLS]位置的隐状态经过两层的MLP降维后得到的打分作为两个句子的相似度得分。同时，尝试了许多的技巧用于进一步提升模型在下游任务上的表现，具体包括：分组分层学习率策略、对抗训练（FGM、PGD）、指数移动平均（EMA）、不同的Dropout策略（R-Drop、Multi-Sample Dropout）、不同层表示Mixup等。训练超参数如下：训练轮数为5，学习率为5e-5，batch size为100，最大句子长度为64。

归因打分部分我们基于梯度计算token embedding各个维度的重要性分数，然后使用L2范数对其进行规约，得到单个字符的重要性分数。具体来说，我们首先计算某个token embedding x关于模型预测类别的对数概率的梯度g，表示模型输出对x各维度的敏感值，接着计算x各维度重要分数g⊙x，其中⊙表示向量的Hadamard积，最后对各个维度的重要性分数进行L2范数规约，即||g⊙x||2，表示该字符的重要性分数。同时我们融合文本之间字符的重叠和句子对之间潜在的关系，得到最终归因的结果。

## 运行说明

### 环境：

```
cuda==11.2
python==3.8.13
GPU型号==GeForce RTX 2080 TI *4张
apex==0.9.10dev
datasets==2.0.0
deepspeed==0.6.0
fairscale==0.4.6
filelock==3.6.0
nltk==3.7
numpy==1.22.3
packaging==21.3
scikit_learn==1.0.2
torch==1.7.1
tqdm==4.63.0
transformers==4.17.0
pandas==1.4.1
```

详情参考 requirments.txt

tips：

1. pytorch请安装GPU版本。

2. 如果apex安装失败会导致运行报错，因为pip直接安装的apex和NVIDIA的apex库不是同一个库，我们需要的是NVIDIA的apex库。
3. 解决方法：安装NVIDIA的apex库，命令如下：

```
git clone https://github.com/NVIDIA/apex
cd apex
pip3 install -v --no-cache-dir ./
```

### 数据简介：

在后训练部分，我们使用了互联网中文相似度数据：[Link](https://github.com/liucongg/NLPDataSet)

在微调部分，我们使用赛题提供的 lcqmc 数据

### 模型简介：

我们使用 huggingface 上的 [chinese_roberta_wwm_ext_large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)

然后经过后训练得到我们的微调使用的模型，下载链接位于：[xinzhicup_chinese_roberta_wwm_ext_large_post_training_50ep_all](https://huggingface.co/Wanfq/xinzhicup_chinese_roberta_wwm_ext_large_post_training_50ep_all) ，下载后请把模型重命名为 “chinese_roberta_wwm_ext_large_50ep_all”，并放在如下目录：“final_project/similarity_v0/posttrain_models/chinese_roberta_wwm_ext_large_50ep_all”

### 代码运行：

```
首先安装好对应环境，进入 final_project 目录
run.sh
最终的结果位置如下：
final_project/similarity_v1/submit/baseline_LCQMC_transfer_aug_one-fold_posttrain_models_chinese_roberta_wwm_ext_large_50ep_all_bs25_accumulate5_lr5e-05_epoch5.0_LINEAR-scheduler_fgm_cls_th_0.97_1_test_b/sim_rationale_f1_f2_f3.zip
```



