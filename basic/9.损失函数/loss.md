# 损失函数

## 1. 交叉熵
语言模型训练最常见的是交叉熵损失。

对于真实标签 $y_t$ ​，若 one-hot 表示为 $q$, 预测分布为 $p$, 则
$$
\mathcal{L}_t = -\sum_{i=1}^{V} q_i \log p_i
$$

因为 $q$ 是 one-hot，只在真实类别 $y_t$ 处为 1，所以可简化为

\\
$$
\mathcal{L}_t = -\log p(y_t \mid x_{<t})
$$

整个序列的平均

$$
\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\log P(x_t \mid x_{<t})
$$

### 1.2困惑度
常用评价指标:
$$ PPL=\exp(\mathcal{L})
$$

- 计算困惑度的参考: 
https://huggingface.co/docs/transformers/perplexity#example-calculating-perplexity-with-gpt-2-in--transformers