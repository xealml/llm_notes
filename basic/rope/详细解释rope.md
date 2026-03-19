```python
把下面的代码转换成原始的公式，方便我理解
def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)

freqs_cis = precompute_freqs_cis()
q_pe = apply_rotary_emb(q_pe, freqs_cis)
```


下面我把这段代码尽量翻译成“原始数学公式 + 直觉解释”。

你可以把它分成两部分看：

1. `precompute_freqs_cis(...)`：先把每个位置要用的旋转角度算好  
2. `apply_rotary_emb(...)`：把输入向量按二维一组做旋转

---

# 一、RoPE 的核心公式

Rotary Positional Embedding 的本质就是：

> 对向量的每一对维度，按位置 `t` 旋转一个角度。

如果某一对维度是：

\[
(x_{2i}, x_{2i+1})
\]

把它看成一个复数：

\[
z_i = x_{2i} + j x_{2i+1}
\]

其中 \(j^2 = -1\)。

那么在位置 \(t\) 上，RoPE 就是把它乘上一个复指数：

\[
z_i' = z_i \cdot e^{j \theta_{t,i}}
\]

其中：

\[
\theta_{t,i} = t \cdot \omega_i
\]

这里 \(\omega_i\) 是第 \(i\) 个频率。

展开后：

\[
e^{j\theta} = \cos\theta + j\sin\theta
\]

所以：

\[
z_i' = (x_{2i} + jx_{2i+1})(\cos\theta_{t,i} + j\sin\theta_{t,i})
\]

对应到实数形式就是一个二维旋转：

\[
\begin{pmatrix}
x'_{2i} \\
x'_{2i+1}
\end{pmatrix}
=
\begin{pmatrix}
\cos\theta_{t,i} & -\sin\theta_{t,i} \\
\sin\theta_{t,i} & \cos\theta_{t,i}
\end{pmatrix}
\begin{pmatrix}
x_{2i} \\
x_{2i+1}
\end{pmatrix}
\]

也就是：

\[
x'_{2i} = x_{2i}\cos\theta_{t,i} - x_{2i+1}\sin\theta_{t,i}
\]

\[
x'_{2i+1} = x_{2i}\sin\theta_{t,i} + x_{2i+1}\cos\theta_{t,i}
\]

这就是 RoPE 最原始的公式。

---

# 二、`precompute_freqs_cis` 对应的公式

---

## 1. 频率 \(\omega_i\) 的定义

代码里这句：

```python
freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
```

对应公式：

设

- 旋转维度为 \(d = \text{dim}\)
- 第 \(i\) 个二维对，\(i = 0,1,\dots,\frac d2-1\)

那么它的频率是：

\[
\omega_i = \frac{1}{\text{base}^{\frac{2i}{d}}}
\]

因为 `torch.arange(0, dim, 2)` 取的是：

\[
0,2,4,\dots,d-2
\]

所以对应就是 \(\frac{2i}{d}\)。

这和标准 Transformer 位置编码的频率设计非常像。

---

## 2. 每个位置的旋转角

代码里：

```python
t = torch.arange(seqlen)
freqs = torch.outer(t, freqs)
```

就是在算外积。

设：

- 位置 \(t = 0,1,\dots,L-1\)
- 频率索引 \(i = 0,1,\dots,\frac d2-1\)

那么得到的角度矩阵是：

\[
\theta_{t,i} = t \cdot \omega_i
\]

也就是说，对于每个位置 \(t\)，每一对维度 \(i\) 都有自己的旋转角。

---

## 3. 复数形式的旋转因子

代码里：

```python
freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
```

`torch.polar(r, theta)` 表示生成复数：

\[
r(\cos\theta + j\sin\theta)
\]

这里半径 \(r=1\)，所以：

\[
\text{freqs\_cis}_{t,i} = e^{j\theta_{t,i}} = \cos(\theta_{t,i}) + j\sin(\theta_{t,i})
\]

也就是：

\[
\text{freqs\_cis}_{t,i} = \cos(t\omega_i) + j\sin(t\omega_i)
\]

这就是预计算出来的旋转表。

---

# 三、长上下文扩展那段在干什么

代码里这段：

```python
if seqlen > args.original_seq_len:
    low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
    smooth = 1 - linear_ramp_factor(low, high, dim // 2)
    freqs = freqs / factor * (1 - smooth) + freqs * smooth
```

这是对 RoPE 频率做“平滑缩放”，目的是支持更长上下文。

---

## 1. 原始问题

标准 RoPE 在训练时通常只在某个原始长度 \(L_0\) 上训练。  
如果推理时长度变成更长 \(L\)，高频分量可能旋转太快，泛化变差。

所以这里对频率做了一种修正。

---

## 2. 修正后的频率

代码等价于：

\[
\omega_i' =
\omega_i \cdot s_i
+
\frac{\omega_i}{\text{factor}} \cdot (1-s_i)
\]

其中 \(s_i = \text{smooth}_i\)。

更常见地写成：

\[
\omega_i' = \omega_i \left(s_i + \frac{1-s_i}{\text{factor}}\right)
\]

这里：

- \(s_i \approx 1\) 的维度：基本保持原频率
- \(s_i \approx 0\) 的维度：频率缩小到 \(\omega_i/\text{factor}\)

也就是说：

> 一部分维度保持原来的旋转速度，一部分维度降低旋转速度。

这样在长上下文下不容易“转过头”。

---

## 3. `smooth` 的作用

`smooth` 是一个从 1 平滑过渡到 0 的函数。

可以理解成：

\[
s_i =
\begin{cases}
1, & i \text{ 在低维区域} \\
\text{线性过渡}, & i \text{ 在中间区域} \\
0, & i \text{ 在高维区域}
\end{cases}
\]

所以修正频率不是硬切换，而是平滑混合。

---

## 4. correction range 那两个函数

### `find_correction_dim(...)`

```python
return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))
```

这个公式是在反推：

> 哪些维度的频率，在原始最大长度 `max_seq_len` 内会转 `num_rotations` 圈？

因为一维对 \(i\) 在长度 \(L\) 内累计旋转角约为：

\[
L \cdot \omega_i
\]

转的圈数约为：

\[
\frac{L\omega_i}{2\pi}
\]

令它等于某个给定旋转圈数 \(r\)：

\[
\frac{L\omega_i}{2\pi} = r
\]

代入

\[
\omega_i = \frac{1}{\text{base}^{2i/d}}
\]

得到：

\[
\frac{L}{2\pi \cdot \text{base}^{2i/d}} = r
\]

解出 \(i\)：

\[
\frac{2i}{d}\log(\text{base}) = \log\left(\frac{L}{2\pi r}\right)
\]

所以：

\[
i = \frac{d}{2} \cdot \frac{\log\left(\frac{L}{2\pi r}\right)}{\log(\text{base})}
\]

代码里的写法和这个是等价的。

它是在找：  
“从哪些维度开始，频率太高，需要修正”。

---

# 四、`apply_rotary_emb` 的数学形式

现在看第二个函数。

---

## 1. 把最后一维两两配对

代码：

```python
x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
```

假设输入：

\[
x \in \mathbb{R}^{B \times T \times H \times d}
\]

其中最后一维 \(d\) 是偶数。

那么它会把每两个元素组成一个复数：

\[
\tilde{x}_{b,t,h,i} = x_{b,t,h,2i} + j x_{b,t,h,2i+1}
\]

于是：

\[
\tilde{x} \in \mathbb{C}^{B \times T \times H \times d/2}
\]

---

## 2. 广播旋转因子

代码：

```python
freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
```

意思是把 `freqs_cis` reshape 成：

\[
1 \times T \times 1 \times d/2
\]

这样它就能对 batch 和 head 维自动广播。

所以对每个 batch、每个 head、每个位置 \(t\)、每个频率 \(i\)，都乘以同一个旋转因子：

\[
e^{j\theta_{t,i}}
\]

---

## 3. 复数乘法实现旋转

代码：

```python
y = torch.view_as_real(x * freqs_cis).flatten(3)
```

对应公式：

\[
\tilde{y}_{b,t,h,i} = \tilde{x}_{b,t,h,i} \cdot e^{j\theta_{t,i}}
\]

也就是：

\[
\tilde{y}_{b,t,h,i}
=
\left(x_{b,t,h,2i} + j x_{b,t,h,2i+1}\right)
\left(\cos\theta_{t,i} + j\sin\theta_{t,i}\right)
\]

展开后：

\[
y_{b,t,h,2i} = x_{b,t,h,2i}\cos\theta_{t,i} - x_{b,t,h,2i+1}\sin\theta_{t,i}
\]

\[
y_{b,t,h,2i+1} = x_{b,t,h,2i}\sin\theta_{t,i} + x_{b,t,h,2i+1}\cos\theta_{t,i}
\]

最后再把复数拆回实数对。

---

# 五、把整段代码合成一个总公式

你最后写的是：

```python
freqs_cis = precompute_freqs_cis()
q_pe = apply_rotary_emb(q_pe, freqs_cis)
```

它整体上就是：

---

## 1. 先定义频率

对于第 \(i\) 个二维对子空间：

\[
\omega_i = \frac{1}{\text{base}^{2i/d}}
\]

如果开启长上下文修正，则变成：

\[
\omega_i' = \omega_i \left(s_i + \frac{1-s_i}{\text{factor}}\right)
\]

其中 \(s_i\) 是一个平滑权重。

---

## 2. 对位置 \(t\) 计算旋转角

\[
\theta_{t,i} = t \cdot \omega_i'
\]

---

## 3. 对 query 的每一对维度做旋转

如果：

\[
q^{(i)}_{t} =
\begin{pmatrix}
q_{t,2i} \\
q_{t,2i+1}
\end{pmatrix}
\]

那么 RoPE 后：

\[
\hat{q}^{(i)}_t =
\begin{pmatrix}
\cos\theta_{t,i} & -\sin\theta_{t,i} \\
\sin\theta_{t,i} & \cos\theta_{t,i}
\end{pmatrix}
q^{(i)}_t
\]

也就是：

\[
\hat{q}_{t,2i} = q_{t,2i}\cos\theta_{t,i} - q_{t,2i+1}\sin\theta_{t,i}
\]

\[
\hat{q}_{t,2i+1} = q_{t,2i}\sin\theta_{t,i} + q_{t,2i+1}\cos\theta_{t,i}
\]

---

# 六、为什么这样做能编码相对位置

这是 RoPE 最漂亮的地方。

假设 query 在位置 \(m\)，key 在位置 \(n\)。

经过旋转后，它们的内积里会自然出现：

\[
(m-n)\omega_i
\]

也就是说 attention 分数依赖的是**相对位置差** \(m-n\)，而不是绝对位置本身。

这是 RoPE 比普通绝对位置编码更适合 attention 的关键原因。

---

# 七、最直白的理解

你可以把这段代码理解成：

### 对每个 token 的一部分向量 `q_pe`
- 不再直接原样使用
- 而是把它拆成很多二维小平面
- 每个小平面按“当前位置”旋转一个角度
- 不同二维平面的旋转速度不同

于是：

- 位置越靠后，旋转角越大
- 不同维度对位置的敏感程度不同
- query/key 的点积就会携带相对位置信息

---

如果你愿意，我下一步可以继续帮你做两件事里的任意一个：

1. **把这段代码写成极简 NumPy / PyTorch 版本**，只保留核心公式  
2. **推导为什么 RoPE 后 \(QK^T\) 会变成相对位置形式**，这个对真正理解 RoPE 很关键