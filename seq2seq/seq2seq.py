import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


# nn.modules 상속
# attention은 query를 변환하는 방법인 linear transformation을 학습함
class Attention(nn.modules):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        # linear transformation
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        # query와 key를 곱한 것에다가 softmax 적용
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h_enc, h_t_dec, mask=None):
        # |h_enc| = (batch_size, length, hidden_size)
        # |h_t_dec| = (batch_size, 1, hidden_size)
        # |mask| = (batch_size, length)

        # linear tranform을 통한 Query 생성
        query = self.linear(h_t_dec)
        # Query에 encoder의 hidden state 행렬곱 진행함으로써 각 단어(Key)의 가중치(유사도) 벡터 return
        # torch.bmm : Batch Matrix Multiplication
        # transpose(1, 2) : batch를 가리키는 index인 0을 제외하고 row(0)와 col(1)만 transpose
        weight = torch.bmm(query, h_enc.transpose(1, 2))

        if mask is not None:
            # masked_fill : mask에 True가 있는 위치(<PAD>의 위치에 True가 있음)에 음의 무한대로 치환
            weight.masked_fill_(mask.unsqueeze(1), -float('inf'))
        weight = self.softmax(weight)
        context_vector = torch.bmm(weight, h_enc)
        return context_vector


# nn.modules 상속
class Encoder(nn.modules):
    def __init__(self, wordvec_dim, hidden_size, n_layers=4, dropout_p=.2):
        super(Encoder, self).__init__()

        self.rnn = nn.LSTM(
            wordvec_dim,
            int(hidden_size / 2),
            num_layers=n_layers,
            dropout=dropout_p,
            # encoder의 경우 bidirectional LSTM 사용가능
            bidirectional=True,
            batch_first=True
        )

    def forward(self, emb):
        # isinstance() : emb(embedding tensor)가 tuple인지 확인
        if isinstance(emb, tuple):
            x, lengths = emb
            x = pack(x, lengths.tolist(), batch_first=True)

            # pack_padded_sequence : Packs a Tensor containing padded sequences of variable length.
            # a = [torch.tensor([1,2,3]), torch.tensor([3,4])] -> tensor의 list
            # b = torch.nn.utils.rnn.pad_sequence(a, batch_first = True)

            # 이 때, b는 다음과 같다
            # tensor([[1, 2, 3],
            #         [3, 4, 0]])
            # (padding 됨, 여기서의 0은 pad)
            # 따라서 rnn에 들어갈때는 (1, 3), (2, 4), (3, 0)끼리 각각의 cell로 들어가게 됨

            # torch.nn.utils.rnn.pack_padded_sequence(b, batch_first = True, lengths=[3, 2])
            # >>> PackedSequence(data=tensor([1, 3, 2, 4, 3]), batch_sizes=tensor([2, 2, 1]), sorted_indices=None, unsorted_indices=None) -> 추후에 sort 부분 참고할 것

            # >>> PackedSequence(data=tensor([1, 3, 2, 4, 3]), batch_sizes=tensor([2, 2, 1])
            # >>> (tensor 안의 실제 데이터 값(pad가 없는 상태), 각 timestep마다의 sample 개수)의 tuple 형태로 반환됨

        else:
            x = emb

        y, hidden = self.rnn(x)
        # y의 경우에는 전체 timestep의 마지막 layer의 hidden state들
        # |y| = (batch_size, length, hidden_size)
        # h의 경우에는 마지막 timestep의 hidden state와 cell state의 tuple로 이루어짐, 따라서 h[0]이 hidden state
        # |h[0]| = (num_layers * 2, batch_size, hidden_size / 2)

        if isinstance(emb, tuple):
            y, _ = unpack(y, batch_first=True)

        return y, hidden