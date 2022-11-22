# 시퀀스-투-시퀀스(Sequence-to-Sequence, seq2seq)

시퀀스-투-시퀀스(Sequence-to-Sequence, seq2seq)는 입력된 시퀀스로부터 다른 도메인의 시퀀스를 출력하는 모델이다.  
기계번역, 챗봇, text summarization 등 다양한 분야에서 사용되지만, 모델 자체의 이해를 돕기 위해 아래의 설명에서는 번역 task를 기반으로 설명해보겠다.

![ 1jpg](https://user-images.githubusercontent.com/74291999/203123223-3c51d355-4e7f-4df2-b746-9030edd2c6db.jpg)  
sequence to sequence는 크게 세 가지 부분으로 나뉜다.

- Encoder
- Decoder
- Generator

각자의 요소들이 어떤 기능을 하는지 차근차근 알아보자

## Encoder
encoder는 입력되는 sequence의 정보를 최대한 보존하도록 압축을 진행하는 역할을 한다.  
**즉, 문장의 의미가 잘 담겨있는 상태로 높은 차원에 있던 data를 낮은 차원의 latent space(잠재 공간)에 투영시키는 것이다.**  
seq2seq에서의 encoder는 입력을 받은 뒤, 정보를 보존하도록 압축을 진행하고, 이렇게 압축한 정보인 context vector를 decoder에 넘겨주는 역할을 한다.  
더 자세한 이해를 위해 Encoder를 수식과 함께 살펴보자 <br>

![1](https://user-images.githubusercontent.com/74291999/203216916-aeec6cea-6485-4ea3-9f98-4752b389de58.png)


Dataset은 $x$와 $y$의 문장 pair로 이루어져 있고, (번역 task에서는 $x$가 한국어 문장, $y$가 영어 문장이라고 가정하자)   
문장 $x$는 $m$개의 단어,  문장 $y$는 $n$개의 단어로 이루어져있다고 한다.  
단, 이때 문장 $y$의 시작과 끝은 각각 <BOS>, <EOS> 토큰으로 이루어져 있다.  
이때, 각 문장들의 shape은 다음과 같다
<br>

![2](https://user-images.githubusercontent.com/74291999/203216913-26025c04-e4f6-4656-a8d5-17de2cf74af3.png)


일단, encoder에는 $x$만 입력으로 들어가기 때문에, $x$가 encoder에 들어가는 과정을 수식으로 살펴보겠다.  
<br>
![3](https://user-images.githubusercontent.com/74291999/203216911-cf639043-bdc4-4543-9542-109c523934a2.png)



먼저, encoder의 input인 문장 $x_t$가 encoder의 embedding layer를 통과한다 (이 때의 shape은 $(batch\_size, 1, embedding\_size)$)    
이후, embedding layer를 통과한 input은 이전 timestep($t-1$)의 hidden state와 함께 RNN layer의 input으로 들어가게 된다.  
이렇게 현재 timestep($t$)의 hidden state를 구하게 된다. (이 때의 shape은 $(batch\_size, 1, hidden\_size)$)  
결론적으로, $h^{enc}_{1:m}$은  $(batch\_size, m, hidden\_size)$의 shape을 가지게 된다  
그런데, 만약 encoder가 bidirectional RNN을 사용하게 된다면 다음과 같은 shape을 가지게 된다.  <br>

![4](https://user-images.githubusercontent.com/74291999/203216910-fa6b4561-1ddb-4521-a8bb-e8cd04e259f8.png)


## Decoder
decoder는 encoder로 압축된 정보를 입력되는 sequence와 같아지도록 압축 해제하는 역할이다.  
**즉, encoder가 압축한 정보를 받아서 단어를 하나씩 뱉어내는 역할이다.**  
seq2seq에서의 decoder는 encoder로부터 정보를 받아온 뒤, encoder의 마지막 hidden state를 decoder의 initial state로 넣어준다.  
수식과 함께 자세히 살펴보도록 하겠다.  

![5](https://user-images.githubusercontent.com/74291999/203216906-cb1080b8-7f59-4249-bcbc-f4945351a676.png)
<br>


우선, encoder 설명때와 마찬가지로 Dataset은 $x$와 $y$의 문장 pair로 이루어져 있고, (번역 task에서는 $x$가 한국어 문장, $y$가 영어 문장이라고 가정하자) 문장 $x$는 $m$개의 단어, 문장 $y$는 $n$개의 단어로 이루어져있다고 한다.


![6](https://user-images.githubusercontent.com/74291999/203216901-e1b6a6c1-768e-4f5b-bc1f-3c8b10e3e836.png)


이전 timestep($t-1$)의 decoder의 output인 $\hat{y}_{t-1}$이  embedding layer를 통과한다 (이 때의 shape은 $(batch\_size, 1, embedding\_size))$)  
이후, embedding layer를 통과한 input은 이전 timestep($t-1$)의 hidden state와 함께 RNN layer의 input으로 들어가게 된다.  
이렇게 현재 timestep($t$)의 hidden state를 구하게 된다. (이 때의 shape은 $(batch\_size, 1, hidden\_size)$)   
결론적으로, $h^{dec}_{1:n}$은  $(batch\_size, n, hidden\_size)$의 shape을 가지게 된다.  
**이러한 decoder는 encoder로부터 문장을 압축한 context vector를 바탕으로 문장을 생성하며, auto-regressive task이기 때문에 bi-directional RNN을 사용하지 못한다는 특징이 있다.**

## Generator
Generator는 decoder의 hidden state를 받아 현재 timestep의 출력 token에 대한 확률 분포를 반환하는 역할을 한다.

수식과 함께 자세히 알아보도록 하겠다.
![7](https://user-images.githubusercontent.com/74291999/203216896-990b026c-c525-4944-8db6-3e88581bd420.png)
encoder와 decoder때와 마찬가지로 dataset은 동일하다.  
Dataset은 $x$와 $y$의 문장 pair로 이루어져 있고, (번역 task에서는 $x$가 한국어 문장, $y$가 영어 문장이라고 가정하자) 문장 $x$는 $m$개의 단어,  문장 $y$는 $n$개의 단어로 이루어져있다.  
<BOS>와 <EOS>는 decoder에만 존재하는 token으로, <BOS>의 경우 decoding(문장 생성)을 시작하는 신호이며, <EOS>이후 순차적으로  decoding(문장 생성)을 진행하다 <EOS>가 출력되게 되면 decoding이 끝났다는 뜻으로, decoding이 종료되게 된다.  

![8](https://user-images.githubusercontent.com/74291999/203216892-fb5653cb-4ba3-4c94-a4e8-80ec67749b00.png)
generator의 경우, decoder의 각 timestep별 output인 $h^{dec}_t$를 입력으로 받는다. 위에서 언급한것처럼, 이전 timestep($t-1$)의 decoder의 output인 $\hat{y}_{t-1}$이  embedding layer를 통과한다 (이 때의 shape은 $(batch\_size, 1, embedding\_size))$) 이후, embedding layer를 통과한 input은 이전 timestep($t-1$)의 hidden state와 함께 RNN layer의 input으로 들어가게 된다. 이렇게 현재 timestep($t$)의 hidden state를 구하게 된다. (이 때의 shape은 $(batch\_size, 1, hidden\_size)$)  

![9](https://user-images.githubusercontent.com/74291999/203216887-79776fe2-92ee-4134-a27f-763a491ebc7d.png)  
이후 decoder의 output을 받아와서 linear layer를 통과시킨 이후 softmax를 적용시켜 단어의 확률 분포를 반환한다.  
즉, 현재 timestep의 단어를 예측하기 위해, 현재 timestep의 결과물을 vocab($V$) 안의 단어 별 확률값로 변환해주는 것이다.  
따라서, linear layer는 decoder의 output($\text{1, hidden size}$)을 vocab의 size($|V|$)로 변환해주기 때문에 $(\text{hidden size, |V|})$의 shape을 가지게 된다.