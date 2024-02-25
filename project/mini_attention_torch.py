import torch
import torch.nn as nn


################## 정의 ###################

class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, inputs):
        # inputs의 차원: (batch_size, seq_len, input_dim)
        # hidden_state의 차원: (batch_size, hidden_dim)
        hidden_state = torch.tanh(self.linear_in(inputs))
        attention_weights = self.linear_out(hidden_state)
        attention_weights = torch.softmax(attention_weights.squeeze(-1), dim=-1)
        # 각 시퀀스 위치에 대한 가중치를 곱하여 어텐션된 표현 계산
        attention_output = torch.sum(inputs * attention_weights.unsqueeze(-1), dim=1)
        return attention_output, attention_weights

################## 사용 ####################

input_dim = 10  # 입력 특징의 차원
seq_len = 20    # 시퀀스 길이
hidden_dim = 64 # 어텐션 특징 공간의 차원

# 모델 인스턴스화
attention_model = Attention(input_dim, hidden_dim)

# 임의의 입력 생성
batch_size = 32
inputs = torch.randn(batch_size, seq_len, input_dim)

# 어텐션을 사용하여 출력 계산
attention_output, attention_weights = attention_model(inputs)

print("Attention Output Shape:", attention_output.shape)
print("Attention Weights Shape:", attention_weights.shape)