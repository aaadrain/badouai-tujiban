# ReadMe
>理解demo中的训练过程，修改正样本生成条件，完成训练和预测
## 第一次完成 work.py
### 1. 主要学习内容：一个nlp简单文本分类模型。   
    - 其中涉及到**词嵌入**->**线性层**->**dropout**->**激活层**->**池化层**->**线性层**->**分类**   
        - 词嵌入的初始化需要先输入形状（词表长度，词表维度）
        - 线性层:需要两个参数，输入维度和输出维度
        - dropout 随机丢弃神经元，其实就是神经元权重置为零
        - 激活层：使用激活函数对权重值进行变换
        - 池化层：作为采样的一种手段。最大值采样或者均值采样
        - 分类层：其实也是线形层。主要是将最后的结果输出到满足类别数的维度中
  


### 2. 主要是使用 **numpy** 复现linear与RNN   

 - linear
      ```math
      y = x·A.T + b
      ```
     - 学习了线性相乘的计算方法 X*A.T，其实权重乘input的转置
 - RNN
      ```
       h_t = tanh(W_ih·x_t + b_ih + W_hh·h_(t-1) + b_hh)
      ```
     - 已知当输入是多维多列的数组
     - 有两个权重矩阵 一个时候 *W* 另一个是 *U* 在公式中是*i*
     - 是一个递归操作
     - 下一次的输入ht是这一次的h_next   每一次的h的输入是上一次的输出状态，再加上新的x
      ```python    
        h_t = np.zeros((self.hidden_size)) # 一开始是默认为零的初始权重
        output = []
        for xi in x:
            print('xi',xi,'xi.T',xi.T)
            ux = np.dot(self.W_ih, xi)
            wh = np.dot(self.W_hh, h_t)
            h_t = np.tanh(ux + wh)
            output.append(h_t)
      ```   
      
### 3. 复现CrossEntropy:   
   #### 交叉熵是多分类的损失计算.

```math
     
  \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                 = -x[class] + \log\left(\sum_j \exp(x[j])\right)
```
```math
H(p,q) = -sum( p(x) * log q(x) )
```
   - 其中需要将true类别转成one-hot编码
```python
        out = torch.zeros(3, 3).scatter(dim=1, index=torch.LongTensor(true_array).view(-1, 1),value=1)
```
   - sum 里面的axis keepdims的用法剖析   
       - such as : shape = (2,6)
       - axis = i，则矩阵运算沿着第i个下标变化的方向进行操作 **shape 在最后的就是对应最内层的**
           - axis=1 ,就是指shape里面最后一个维度, 即6这个维度
       - **axis的意思是移除shape里面的哪个维度。**
           - axis=1 就是值我要把6这个维度移除.保留2,这个维度