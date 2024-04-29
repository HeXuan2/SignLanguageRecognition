import torch
import torch.nn as nn
import torch.nn.functional as F

"""
这个自动编码器模型由两部分组成：编码器（encoder）和解码器（decoder）。编码器部分包括几个线性层（nn.Linear）和 Tanh 激活函数，用于将输入数据压缩为具有 OUTPUT_SIZE 个特征的表示；解码器部分也包括几个线性层和 Tanh 激活函数，用于将压缩后的表示解压缩为原始维度，并通过 Sigmoid 函数将输出限制在 (0, 1) 之间。
在 forward 方法中，输入 x 经过编码器和解码器的处理得到编码后的表示 encoded 和解码后的重构结果 decoded，然后返回这两部分结果。这样，整个模型就可以实现数据的压缩和解压缩功能。
"""
class AutoEncoder(nn.Module):
    def __init__(self, INPUT_SIZE, TIME_STEP, OUTPUT_SIZE):
        super(AutoEncoder, self).__init__()

        # 定义编码器部分
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_SIZE*TIME_STEP, 128),  # 输入层
            nn.Tanh(),  # Tanh激活函数
            nn.Linear(128, 64),  # 隐藏层
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, OUTPUT_SIZE),   # 输出层，将输入数据压缩为具有OUTPUT_SIZE个特征的表示
        )

        # 定义解码器部分
        self.decoder = nn.Sequential(
            nn.Linear(OUTPUT_SIZE, 12),  # 输入层，接收编码器输出作为输入
            nn.Tanh(),
            nn.Linear(12, 64),  # 隐藏层
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, INPUT_SIZE*TIME_STEP),  # 输出层，将压缩后的表示解压缩为原始维度
            nn.Sigmoid(),       # 最终输出经过 Sigmoid 函数，将数值范围限制在 (0, 1) 之间
        )

    def forward(self, x):
        encoded = self.encoder(x)  # 编码器部分的前向传播，将输入编码为低维表示
        decoded = self.decoder(encoded)  # 解码器部分的前向传播，将编码后的表示解码为原始维度
        return encoded, decoded  # 返回编码后的表示和解码后的重构结果
