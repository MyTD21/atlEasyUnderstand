# transformer 
## 关于模拟计算
- 本项目仅作示范，所有模型参数/矩阵未经过训练，用random_matrix随机生成一个矩阵；
- 模拟类翻译的seq2seq任务，源序列是src，目标序列是tgt，output是模型预测的结果；
- 训练阶段，tgt作为监督信号，用于更新loss；推理阶段，tgt就是已经生成的output，自回归生成；
- 通篇例子如下，我 爱 吃 火锅 -> I like to eat hot pot

## Transformer类
### 流程（forward函数）
```mermaid
graph LR
    src词嵌入 --> src位置编码 --> encoders -.-> tgt词嵌入 --> tgt位置编码 --> decoders --> 归一化+投影
```
### 词嵌入(Word Embedding)
- 将文字映射到高维空间，一般一个字/词会对应一个512(d_model)的向量;
- Word Embedding通过查表获取，但是权重也是需要学习的；
  
### 位置编码（PE）
- 目的是给序列加入位置信息，由于输入是 [seq_len, d_model]的向量，需要生成对应size的向量和输入相加；
- 计算方式：

  pe[pos, i] = math.sin(pos / (10000** (2*i / d_model)))
  
  pe[pos, i+1] = math.cos(pos / (10000 **(2*i / d_model)))

  其中pos是词的位置，如例中'吃'这个字，位置是3，则计算'吃'的PE时，这个pos就是3；i是Word Embedding的维度；

### LayerNorm
- 核心思想是对单个样本的特征维度进行归一化，使数据分布更稳定，从而加速训练并提升模型性;
- 单个样本是指，在d_model这个层面算均值方差这些；每个词嵌入都是自己内部归一化，"爱"和"吃"各自算各自的均值，和其他字没关系；
- 文本序列长度不固定，如果在batchNorm计算会受到padding的影响；
- 会在均值方差的基础上，加一点偏移，缩放之类的操作；
  
### encoders
- 一般是多个encoder，encoder1的输出，作为encoder2的输入；
- 公式
  
  y_self = LayerNorm(self_attn(x) + x)
  
  ret = LayerNorm(FF(y_self) + y_self)

### decoders
- 一般是多个decoder，decoder1的输出，作为decoder2的输入；
- 公式
  
  y_self = LayerNorm(self_attn(x, mask) + x)
  
  y_cross = LayerNorm(cross_attn(y_self, enc_out) + y_self)
  
  ret = LayerNorm(FF(y_cross) + y_cross)
  
### 输出归一化+投影
- 对decoder的结果进行归一化，并投影到输出维度，(1, 15, 512) -> (1, 15, 10000)

# rnn
