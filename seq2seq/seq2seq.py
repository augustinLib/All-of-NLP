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
    def __init__(self, wordvec_dim, hidden_size, n_layers = 4, dropout_p = .2):
        super(Encoder, self).__init__()

        self.rnn = nn.LSTM(
            wordvec_dim,
            int(hidden_size / 2),
            num_layers = n_layers,
            dropout = dropout_p,
            # encoder의 경우 bidirectional LSTM 사용가능
            bidirectional=True,
            batch_first = True
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
        

# nn.modules 상속
class Decoder(nn.Module):
    def __init__(self, word_vec_size, hidden_size, n_layers = 4, dropout_p = .2):
        super(Decoder, self).__init__()
        
        self.rnn = nn.LSTM(
            # input feeding을 위해 이전 timestep의 h_tilde concat하기 때문에 input size는 word_vec_size + hidden_size
            word_vec_size + hidden_size,
            hidden_size,
            num_layers = n_layers,
            dropout = dropout_p,
            # decoder는 autoregressive하기 때문에 bidirectional = False
            bidirectional = False,
            batch_first = True
        )
    
    # decoder의 forward는 encoder의 forward와는 다르게 하나의 timestep씩 입력으로 들어옴
    def forward(self, emb_t, h_t_1_tilde, h_t_1):
        # emb_t : 현재 timestep의 embedding vector -> (batch_size, 1, embedding_size)
        # h_t_1_tilde : 이전 timestep의 output -> (batch_size, 1, hidden_size)
        # h_t_1 : 이전 timestep의 hidden state, h_t_1 = 이전 timestep의 hidden state와 cell state가 같이 들어가있을 것
        # 따라서 h_t_1[0]으로 hidden state 추출 -> (n_layers, batch_size, hidden_size)
        
        batch_size = emb_t.size(0)
        hidden_size = h_t_1[0].size(-1)
        
        # 첫 timestep
        if h_t_1_tilde is None:
            # (batch_size, 1, hidden_size) shape으로 0으로 채워진 tensor 생성
            # emb_t.new(shape) : emb_t와 같은 device, 같은 type으로 (shape)의 크기를 가진 텐서 생성
            h_t_1_tilde = emb_t.new(batch_size, 1, hidden_size).zero_()
        
        # 마지막 dim으로 concat -> (batch size, 1, hidden_size + embedding_size)
        x = torch.cat([emb_t, h_t_1_tilde], dim = -1)
        
        y, hidden = self.rnn(x, h_t_1)
        
        return y, hidden
    
    
 # nn.modules 상속   
class Generator(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Generator, self).__init__()
        
        self.output = nn.Linear(hidden_size, output_size)
        # 마지막 차원에 대해 LogSoftmax 적용
        self.softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, x):
        y = self.softmax(self.output(x))
        
        return y
    

# nn.modules 상속   
class Seq2Seq(nn.Module):
    def __init__(self,
                 input_size,
                 word_vec_size,
                 hidden_size,
                 output_size,
                 n_layers = 4,
                 dropout_p = .2
                 ):
        
        self.input_size = input_size
        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        super(Seq2Seq, self).__init__()
        
        # encoder embedding
        self.emb_enc = nn.Embedding(input_size, word_vec_size)
        # decoder embedding
        self.emb_dec = nn.Embedding(output_size, word_vec_size)
        
        self.encoder = Encoder(
            wordvec_dim = word_vec_size,
            hidden_size = hidden_size,
            n_layers = n_layers,
            dropout_p = dropout_p
        )
        
        self.decoder = Decoder(
            word_vec_size = word_vec_size,
            hidden_size = hidden_size,
            n_layers = n_layers,
            dropout_p = dropout_p
        )
        
        self.attention = Attention(hidden_size = hidden_size)
        
        # attention의 output인 context vector와 decoder의 output을 concat한 이후, hidden size로 바꿔줌
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.tanh = nn.Tanh()
        self.generator = Generator(hidden_size = hidden_size, output_size = output_size)
        
    def generate_mask(self, x, length):
        # |x| : (bs, n, |V|)
        # |length| : (bs, ) -> batch 내부 sample별 길이가 들어가있다.
        mask = []
        
        max_length= max(length)
        for len in length:
            if max_length - len > 0:
                # torch.tensor.new_ones() : tensor와 같은 type, 같은 device의 1로 채워진 tensor 생성 
                # Tensor.zero_() → Tensor : tensor 객체를 0으로 채움
                # 문장 앞부분(len : 문장의 길이)은 0으로 채워주고 -> x.new_ones(1, len).zero_()
                # 문장 뒷부분(max_len - len : 최대 문장 길이에 비해서 부족한 문장의 길이 수)은 1로 채워줌
                # 이후 torch.cat(dim = -1) 으로 concat
                    # x = torch.rand(batch_size, N, K) # [M, N, K]
                    # y = torch.rand(batch_size, N, K) # [M, N, K]
                    # output1 = torch.cat([x,y], dim=1) #[M, N+N, K]
                    # output2 = torch.cat([x,y], dim=2) #[M, N, K+K]
                
                mask += [torch.cat([x.new_ones(1, len).zero_(),
                                    x.new_ones(1, (max_length - len))
                                    ],
                                   dim= -1)
                         ]
            
            else:
                # sample 문장의 길이가 max_length와 같으면 모든 mask 0으로 채움
                mask += [x.new_ones(len, 1).zero_()]
                
                
        mask = torch.cat(mask, dim = 0).bool()
        
        return mask
    
    
    # encoder의 마지막 hidden state를 decoder의 첫 hidden state로 넣어줘야함
    # encoder의 hidden state sequential하게 merge
    def merge_encoder_hiddens(self, encoder_hiddens):
        new_hiddens = []
        new_cells = []
        
        # lstm output = (hidden_state, cell_state)
        hiddens, cells = encoder_hiddens
        # |hiddens| : (#layers * 2, batch_size, hidden_size / 2)
        
        for i in range(0, hiddens.size(0), 2):
            new_hiddens += [torch.cat([hiddens[i], hiddens[i+1]], dim=-1)]
            # |new_hiddens| : (batch_size, hidden_size)
            new_cells += [torch.cat([cells[i], cells[i+1]], dim=-1)]
        
        
        new_hiddens, new_cells = torch.stack(new_hiddens), torch.stack(new_cells)
        # torch.stack(tensors, dim=0, *, out=None) → Tensor
        # Concatenates a sequence of tensors along a new dimension.
        # |new_hiddens| : (#layers, batch_size, hidden_size)
        
        return (new_hiddens, new_cells)
    
    
    # encoder의 마지막 hidden state를 decoder의 첫 hidden state로 넣어줘야함
    # encoder의 hidden state parallel하게 merge
    def fast_merge_encoder_hiddens(self, encoder_hiddens):
        h_0_dec, c_0_dec = encoder_hiddens
        # |h_0_dec| : (#layers * 2, batch_size, hidden_size / 2)
        batch_size = h_0_dec.size(1)
        
        h_0_dec = h_0_dec.transpose(0, 1).contiguous().view(batch_size,
                                                            -1,
                                                            self.hidden_size
                                                            ).transpose(0, 1).contiguous()
        # contiguous() : 
        # Tensor.view(*shape) : Returns a new tensor with the same data as the self tensor but of a different shape.
        # reshape()과 다르게 view는 contiguous한 상태에서만 적용 가능
        # |h_0_dec| : (#layers * 2, batch_size, hidden_size / 2)
        # |h_0_dec.transpose(0, 1)| : (batch_size, #layers * 2, hidden_size / 2)
        # |h_0_dec.transpose(0, 1).view(batch_size, -1, self.hidden_size)| : (batch_size, #layers, hidden_size)
        # |h_0_dec.transpose(0, 1).view(batch_size, -1, self.hidden_size).transpose(0, 1)| : (#layers, batch_size, hidden_size)
        
        c_0_dec = c_0_dec.transpose(0, 1).contiguous().view(batch_size,
                                                    -1,
                                                    self.hidden_size
                                                    ).transpose(0, 1).contiguous()
        
        return h_0_dec, c_0_dec
        
    
    
    # input : teacher forcing으로 인해 source text(src)와 target text(tgt)모두 입력받음
    # output : 문장 별 각 timestep별 단어들의 log 확률값
    def forward(self, src, tgt):
        # |src| : (bs, n) ~ (bs, n, |V|)
        # |tgt| : (bs, m) ~ (bs, m, |V|)
        # |output| = (bs, m, |V|)
    
        batch_size = tgt.size(0)
        
        mask = None
        x_length = None
        if isinstance(src, tuple):
            x, x_length = src
            mask = self.generate_mask(x, x_length)
            # |mask| : (batch_size, length)
            
        else:
            x = src
            
        
        if isinstance(tgt, tuple):
            tgt = tgt[0]
            
        emb_enc = self.emb_enc(x)
        # |emb_enc| : (batch_size, length, word_vec_size)
        

        h_enc, h_0_dec = self.encoder((emb_enc, x_length))
        # |h_enc| : (batch_size, length, hidden_size)
        # |h_0_dec| : (n_layers * 2, batch_size, hidden_size / 2)
        
        h_0_dec = self.fast_merge_encoder_hiddens(h_0_dec)
        emb_dec = self.emb_dec(tgt)
        # |emb_dec| : (batch_size, length, word_vec_size)
        
        h_tilde = []
        
        h_t_tilde = None
        decoder_hidden = h_0_dec
        for t in range(tgt.size(1)):
            emb_t = emb_dec[:, t, :].unsqueeze(1)
            # |emb_dec[:, t, :]| : (batch_size, word_vec_size) -> unsqueeze(1)
            # |emb_t| : (batch_size, 1, word_vec_size)
            # |h_t_tilde| : (batch_size, 1, hidden_size)
            
            decoder_output, decoder_hidden = self.decoder(emb_t,
                                                          h_t_tilde,
                                                          decoder_hidden
                                                          )
            
            # |decoder_output| : (batch_size, 1, hidden_size)
            # |decoder_hidden| : (n_layers, 1, hidden_size)
            
            context_vector = self.attention(h_enc, decoder_output, mask)
            # |context_vector| : (batch_size, 1, hidden_size)
            
            h_t_tilde = self.tanh(self.concat(torch.concat([decoder_output,
                                                            context_vector
                                                            ],
                                                            dim=-1)
                                              )
                                  )
            # |h_t_tilde| = (batch_size, 1, hidden_size)
            h_tilde += [h_t_tilde]
        
        h_tilde = torch.cat(h_tilde, dim= 1)
        # |h_tilde| : (batch_size, 1, hidden_size)
        
        y_hat = self.generator(h_tilde)
        # |y_hat| : (batch_size, 1, output_size)
        
        return y_hat
            