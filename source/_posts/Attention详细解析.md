---
title: Attention详细解析
top: false
cover: false
toc: true
mathjax: true
date: 2019-12-08 22:58:22
password:
summary:
tags: 
- Attention
- BERT
- 网络结构
categories:
- NLP
---
# Attention机制理解
## 原文
[Attention机制详解（一）——Seq2Seq中的Attention](https://zhuanlan.zhihu.com/p/47063917)
## Attention提出理由
### 解决痛点
在传统的机器翻译中采用encoder-decoder结构，encoder将输入的句子将其转换为定长的向量，然后decoder将向量转化为目标文字。且通常是encoder将最后一层hidden vector作为decoder的起始，然后通过decoder翻译为其他语言。这其中会由于RNN自身特性带来长程梯度消失和并行化差的问题。其中较长的句子也较难在最后的vector中保存需要的有效信息
### 提出解决方案
模拟人翻译的过程，当翻译部分词时将注意力或者更多的注意力放在需要关注的词上，通过类似于赋权的方式计算每个输入位置j与输出位置的关联性。例如可以计算每个输入位置j和当前输出位置的关联性$e_{tj} = a(s_{t-1}, h_j)$,所以写成向量形式就可以得到$\stackrel{->}{e_t} = (a(s_{t-1}, h_1), ..., a(s_{t-1}, h_T))$ $a$是一种相关性的算符，常见的有点乘形式$\stackrel{->}{e_t}=\stackrel{->}{s_{t-1}}^T\stackrel{->}{h}$,加权点乘$\stackrel{->}{s_{t-1}}^TW\stackrel{->}{h}$, 加和$\stackrel{->}{v}^Ttanh(W_1\stackrel{->}{h} + W_2\stackrel{->}{s_{t - 1}})$,然后$\stackrel{->}{s_{t - 1}}$进行softmax操作将normalize得到attention的分布
### self-attention提出原因
尽可能的去除RNNs网络结构，解决RNN由于其顺序结构进行训练，训练速度会受到约束。在RNN中需要处理对句子中的词一步步地进行顺序处理，并且当它们相距较远时候效果较差。Self-Attention利用了Attention的机制，计算每个单词和其他所有单词之间的关联。可以更好地考虑上下文的信息
### Transformer整体结构解析
使用Multi-head Attention将多个Self-Attention结构结合，每个head会学习到不同的表征，给模型更大的容量
### Self-Attention详细解析
Self-Attention基本结构如下![avatar](./scaled_dot_product_attention.jpg)
#### 对于Self-Attention的利用
对于Self-Attention来说使用来自一个输入的Q(Query)、K(Key)、V(value)进行计算。首先计算Q与K之间的点乘，然后防止其结果过大，除以一个尺度标度$\sqrt{d_k}$,其中$d_k$为一个query和key向量的维度。再利用Softmax将其结果归一化为概率分布，然后再乘以矩阵V就得到权重求和的表示。该操作表示为$Attention(Q, K, V)=softmax(QK^T\div\sqrt{d_k})V$,其中Q,K,V都是通过输入向量进行矩阵运算得到。有一个可视化较好的[解释](https://zhuanlan.zhihu.com/p/47282410)。需要注意的点是，在类似于encoder和decoder的第一层中q,k,v都是使用来自前一层的decoder的输出，但是在decoder的第二层使用的是来自q是来自encoder的输出，k,v是来自decoder的第一层结果。同时在decoder中使用的不是单纯的Multi-Head Attention而是使用了Masked Multi-Head Attention（因为在翻译过程中不知道后面的输入?)。
#### 其他结构
使用了Positional Encoding，该方法主要是将模型没有recurrence和convolution的结构导致没够关于单词在源句子中的位置或绝对的信息，为了让模型更好地学习位置信息的产物，Transformer是使用了三角函数的方式进行encoding。同时在每一步的Multi-Head Attention之后使用了Add和Normanize操作，其中Add表示Residual Connection,该方法是为了解决多层网络训练困难的问题，通过将前一层的信息无差地传递到下一层，可以有效的关注差异部分，这一方法之前在ResNet等图像处理中经常被使用到。而Norm是代表Layer Normalization，该方法通过对层的激活值得归一化，加速模型的训练过程，使得模型可以更快地收敛[Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
## Attention模型的应用
### 自然语言处理
#### 创造新的结构Universal Transformers
[Universal Transformers](https://arxiv.org/pdf/1807.03819.pdf)<br>
该文章结合了Transformer结构和RNN循环归纳的优点，使得Transformer结构能够适用更多自然语言理解的问题。
#### 创造新的预训练模型Bert等
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)<br>
使用双向的Transformer进行预处理，得到包含有上下文信息的表征，根据表征可以fine-tune很多自然语言处理任务，对于GLUE Benchmark(主要包含MNLI,RTE：比较两个句子的语义关系，QQP：判别Quora上两个问题相似度，QNLI：问答，SST-2：情感分析，CoLA:语句合理性判别，STS-B, MRPC：句子相似度判别)，SQuAD(问答)，NER（命名实体识别）等都有极大的提高.
#### 文本生成
[Generating Wikipedia by Summarizing Long Sequences](https://arxiv.org/abs/1801.10198.pdf)
### 图像处理及合成
#### Attention利用始祖
[Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044)<br>
利用Attention机制进行Image Caption(将图像翻译为文字表述)
#### 文本合成和超分使用
[Image Transformer](https://arxiv.org/abs/1802.05751)<br>
可以使用Attention机制对图像进行合成，例如将局部图像进行补全，也可以将低分辨率的图像还原高分辨率的图像。同时由于Image Transformer模型训练的稳定性，可能和GAN有抗衡之势
### 其他领域结合
#### 推荐
[Neural Attentive Session-based Recommendation](https://arxiv.org/pdf/1711.04725.pdf)<br>
利用Attention模型处理用户sesstion中的序列信息进行相关推荐
#### 音乐生成
[Generating Long-Term Structure in Songs and Stories]()<br>
使用Attention RNN创作乐曲