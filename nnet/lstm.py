import torch
import torch.nn as nn
import torch.nn.functional as F

# 这个模型包含了一个单向LSTM层以及一个线性输出层。在前向传播中，输入 x 经过LSTM层得到输出 r_out，然后通过线性输出层 self.output 将输出映射到最终输出的维度。最终，模型返回这个输出。


# 定义LSTM的结构
class lstm(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, LAYERS, DROP_RATE):
        super(lstm, self).__init__()

        # 定义LSTM层
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,  # 输入数据的特征维度
            hidden_size=HIDDEN_SIZE,  # 隐层状态的维度
            num_layers=LAYERS,  # LSTM层的层数
            dropout=DROP_RATE,  # 防止过拟合的dropout率
            batch_first=True,  # 输入输出数据格式为(batch, seq_len, feature)
            bidirectional=False  # 使用单向LSTM
        )

        # 定义输出层，将LSTM的输出映射到最终输出的维度
        self.output = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)  # 最后一个时序的输出接一个全连接层

        # 初始化隐层状态和细胞状态
        self.h_s = None
        self.h_c = None

    def forward(self, x):  # x是输入数据集
        r_out, (h_s, h_c) = self.rnn(x)  # LSTM的前向传播
        # 如果不导入h_s和h_c，默认每次都进行0初始化
        # h_s和h_c表示每一个隐层的上一时间点输出值和输入细胞状态
        # h_s和h_c的格式均是(num_layers * num_directions, batch, HIDDEN_SIZE)
        # 如果是双向LSTM，num_directions是2，单向是1

        out = self.output(r_out)  # 将LSTM的输出映射到最终输出的维度
        return out  # 返回模型的输出
