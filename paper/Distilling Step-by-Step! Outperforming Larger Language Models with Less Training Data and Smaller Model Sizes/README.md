# Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes

이번 게시물에서는 ACL 2023에 accept 된, LM의 rationale을 distillation에 활용하여 small model이 less training data 상황에서도 좋은 downsteam task 성능을 가지게끔 한 연구인 Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes 논문에 대해 다뤄보겠다



원문 링크는 아래와 같다

https://arxiv.org/abs/2305.02301



# Introduction

LLM이 제공하는 impressive few-shot 성능에도 불구하고, 이러한 model의 매우 큰 크기로 인해 실제 서비스에서 deploy하기에는 큰 어려움이 존재한다. 그 예시로, 175B(GPT-3의 크기)의 LLM을 serving하기 위해서는 최소 350GB의 GPU memory를 필요로 한다. 게다가, PaLM과 같은 LLM은 그 크기가 175B보다 훨씬 큰 540B의 크기를 가진다. 이는 앞서 말한 GPU 메모리보다 더 많은 자원을 필요로 한다는 것을 의미한다.

이러한 large model deployment의 어려움을 피하기 위해, 많은 product에서는 smaller specialized model을 사용하게 된다. 이러한 smaller model들은 보통 fine-tuning이나 distillation을 통해 학습되곤 한다.

Fine-tuning은 BERT나 T5와 같은 pre-trained smaller model을 downstream human annotated data를 통해 학습시키게 되며, distillation은 똑같이 smaller model을 학습시키긴 하지만, LLM에서 생성된 label들을 통해 학습시킨다는 차이가 존재한다.

그러나 위와 같은 방법들도 한계가 존재한다. Fine-tuning으로 LLM과 비슷한 성능으로 끌어올리기 위해서는 expensive human label이 많이 필요하고, distillation의 경우에도 상당한 양의 unlabeled data가 필요하다는 한계가 존재한다.

따라서 본 논문에서는 이러한 한계를 극복하기 위해 Distilling step-by-step을 제안한다. 저자들은 해당 방법론은 기존 방법론에 비해 더 적은 training data를 요구한다고 말한다.

해당 방법론의 핵심은 LLM을 단순히 noisy label의 출처로 보는 것이 아닌, reasoning할 수 있는 agent로 보는 것이라고 한다. 이에 대해 조금 더 자세히 설명하기 위해, 아래의 예시와 함께 살펴보도록 하겠다.

아래와 같은 질문이 있다고 가정해보자

> Jesse’s room is 11 feet long and 15 feet wide. If she already has 16 square feet of carpet. How much more carpet does she need to cover the whole floor?

LLM의 경우, CoT를 사용하여 아래와 같은 intermediate rationale들을 생성할 수 있다.

> Area = length × width. Jesse’s room has 11 × 15 square feet. (11 × 15) − 16

이러한 rationale들은 Area = length × width 와 같은 relevant task knowledge를 가지고 있으며, 이는 원래 small task-specific model이 학습하기에는 매우 많은 양의 training data를 필요로 하는 knowledge이다.

저자들은 이러한 사실을 바탕으로, 이러한 rationale들을 추출하여 small model로 하여금 rationale prediction task를 추가적으로 수행하게 함으로써 더 많은 information을 학습하게 했다고 한다.

Distilling step-by-step은 task-specific small model이 LLM에 비해 500배 가량 작은 크기를 가지고 있음에도 불구하고 더 좋은 성능을 내게끔 하며, 기존 fine-tuning 방법론과 distillation 방법론보다 더 적은 training data를 요구한다고 말한다. 아래는 이를 나타낸 figure이다.

<img width="491" alt="1" src="https://github.com/augustinLib/All-of-NLP/assets/74291999/edee1f81-c075-42ed-a9d6-8124aaf779bc">

저자들은 770M T5 model로 기존 fine-tuning 방법론에 필요한 data 개수의 80%의 data만으로 540B LLM의 성능을 능가하였다고 한다.

# Distilling step-by-step

본 논문에서 제안하는 Distilling step-by-step은 LLM의 reasoning 능력을 smaller model 학습에 활용하는 방법론이다. 아래는 Distilling step-by-step의 전반적인 과정을 나타낸 figure이다.

<img width="979" alt="2" src="https://github.com/augustinLib/All-of-NLP/assets/74291999/ea1092f6-2f7b-4642-9bf5-6499b79e26ec">

우선, LLM과 unlabeled dataset이 주어지면 LLM으로 하여금 output label과 해당 label에 대한 rationale들을 생성하게끔 한다. 이후, 이렇게 생성된 rationale들을 smaller downstream model 학습에 대한 label로 추가한다. 직관적으로 봤을 때, 이러한 rationale들은 input이 왜 output에 대응하는지에 대한 더 풍부하고 자세한 정보를 제공하며, 간혹 relevant task knowledge를 담고 있기도 하다.

## Extracting rationales from LLMs

본 연구에서는 LLM의 few-shot CoT를 활용한다. 아래의 figure와 같이 살펴보자

<img width="481" alt="3" src="https://github.com/augustinLib/All-of-NLP/assets/74291999/d72a3122-1eec-4749-8466-772175847b6f">

unlabeled dataset $x_i \in D$가 주어지면, 먼저 prompt template $p$를 골라낸다. 각각의 prompt는 $(x^P , r^P , y^P)$ triplet이며, $x^P$는 example input, $y^P$는 이에 대응하는 label이며 $r^P$는 $x^P$가 왜 $y^P$에 대응되는지 설명하는 user-provided rationale이다.

각각의 input $x_i$에 $p$를 붙여준 뒤, 이를 LLM이 각각의 $x_i \in D$에 대한 rationale과 label을 생성할 때의 input으로 사용한다. 즉, few-shot CoT를 통해 unlabeled dataset $x_i \in D$에 대한 rationale과 label을 생성한다는 것이다. 이를 통해, 위 figure에서 $P$는 few-shot CoT 부분에 해당할 것이고, $x_i \in D$는 input에 대응한다는 것을 알 수 있다.

## Training smaller models with rationales

저자들은 Distilling step-by-step에 대해 이야기하기에 앞서, learning task-specific model 관련된 다른 연구들 관련하여 언급한다.

### Standard finetuning and task distillation

task-specific model을 학습하는 가장 일반적인 방법은 supervised data를 이용하여 pre-trained model을 finetuning하는 것이다. 만약 human-annotated label이 부족하다면, LLM을 teacher model로 활용하여 pseudo noisy training label을 사용하는 task-specific distillation을 사용해왔다.

두 방법 모두 smaller model $f$는 아래의 label prediction loss를 minimize하도록 학습된다.

<img width="481" alt="4" src="https://github.com/augustinLib/All-of-NLP/assets/74291999/d580ede9-9367-4ef5-b52b-6af20f0ef0a5">

여기서 $\ell$은 cross entropy loss를 의미한다. 또한 설명의 편의를 돕기 위해 $\hat{y}_i$는 fine-tuning에서의 human annotated label과 distillation에서의 LLM predicted label 둘 모두를 의미한다.

### Multi-task learning with rationales

$x_i$과 $\hat{y}_i$ 사이에 보다 명시적인 연결을 만들기 위해, 저자들은 extracted rationales $\hat{r}_i$를 additional supervision으로 사용하였다.

rationale을 downstream model의 학습 과정에 사용한 연구들은 존재해왔는데, 그 중 하나는 $\hat{r}_i$을 additional input으로 사용한 방법론이다. 즉, 아래와 같이 학습이 진행되었다

<img width="481" alt="5" src="https://github.com/augustinLib/All-of-NLP/assets/74291999/836b416b-386d-4009-9e3c-c87384337852">

그러나 이러한 방법의 경우  LLM이 rationale을 만들어야 smaller model이 prediction을 만들어낼 수 있다. 즉, smaller model을 deploy할 때 여전히 LLM이 필요하다는 것이다.

본 연구에서는 rationale을 additional model input으로 사용하는 방법 대신, rationale을 predict하는 task를 추가하여, 결과적으로 multi-task problem으로 학습하는 방법을 채택하였다. 즉, input $x_i$를 사용하여 label뿐만 아니라 rationale도 predict하게끔 하는 것이다. 이는 아래와 같이 표현된다

<img width="481" alt="6" src="https://github.com/augustinLib/All-of-NLP/assets/74291999/100b8534-2009-4f3d-87a9-b77f0ee35a60">

$\mathcal{L}*{\text{label}}$은 앞서 봤었던 equation 1이며, $\mathcal{L}*{\text{rationale}}$은 아래와 같다

<img width="481" alt="7" src="https://github.com/augustinLib/All-of-NLP/assets/74291999/d1c878f2-60d7-47ea-a6cc-b9934fbb3edb">

이러한 rationale generation loss는 smaller model로 하여금 intermediate reasoning step을 생성할 수 있게 학습시키는 역할을 수행한다. 이는 결과적으로 model이 더 좋은 label을 predict하게끔 해준다.

이러한 방법은 앞서 살펴봤던 방법론과는 다르게, smaller model을 deploy할 때 LLM이 필요없다는 장점이 존재한다.

추가적으로, 이러한 Distilling step-by-step 방법론을 활용하여 smaller model을 학습할 때, input example에 대해 [label], [rationale]과 같은 task prefix를 prepend해준다. 즉, rationale을 predict할때는 [rationale] prefix를 prepend해주고, label을 predict할때는 [label]을 prepend해주는 것이다

# Experiments

저자들은 Distilling step-by-step의 성능을 측정해보기 위해 다양한 실험을 진행하였다. 이를 통해 아래와 같은 결과를 얻을 수 있었다고 한다

- 기존 fine-tuning 방법론과 distillation 방법론과 비교했을때, training example이 더 적었음에도 불구하고 더 좋은 성능을 냄
- LLM에 비해 작은 크기의 model을 활용했음에도 불구하고, LLM의 성능을 능가함
- Training example의 수와 model size의 minimum requirement를 비교했을 때, Distilling step-by-step이 LLM을 능가함

추가적으로, Distilling step-by-step 내에서 구성 요소를 다르게 했을때의 성능 변화를 측정하기 위한 비교 실험도 진행하였다고 한다.

### Setup

본 연구의 실험에서 LLM으로는 540B 크기의 PaLM model을 사용했으며, task-specific downstream model로는 pre-trained T5를 사용했다고 한다. CoT prompting은 기존 연구에서 사용했던 방법을 따랐으며, 새로운 dataset에 대한 자체적인 CoT example을 선정했다고 한다.

### Datasets

성능 측정을 위해  아래와 같은 benchmark dataset을 사용했다고 한다.

- e-SNLI
- ANLI
- CQA
- SVAMP

## Reducing training data

저자들은 reducing training data측면에서 Distilling step-by-step을 아래 상황에 따라 두 setting과 비교했다고 한다

- Standard finetuning → when human-labeled examples are available
- Standard task distillation → when only unlabeled examples are available

또한, 비교를 위한 실험을 진행할 때, task-specific model은 220M size의 T5-Base로 고정하였고, 해당 T5-Base model에 Distilling step-by-step, standard finetuning, standard task distillation을 적용하였으며, 각각의 방법론에 사용 가능한 training example의 개수를 다르게 하면서 실험을 진행했다고 한다.

### Distilling step-by-step outperforms standard finetuning with much less labeled examples

우선, human-labeled example이 available한 경우에, standard finetuning과 distilling step-by-step을 비교한 결과이다

아래의 figure와 함께 결과를 살펴보자

<img width="984" alt="8" src="https://github.com/augustinLib/All-of-NLP/assets/74291999/0c3fd2ef-66ed-4fbd-adf9-a5473c6a79c8">

Distilling step-by-step과 standard fine-tuning 모두 training example의 수가 증가할수록 꾸준히 성능 증가가 이루어지는 것을 확인할 수 있다. 또한, 모든 경우에서 distilling step-by-step이 standard fine-tuning보다 더 좋은 성능을 내는 것을 확인할 수 있다. 특히, standard fine-tuning으로는 full training set을 사용해야만 달성할 수 있는 성능을 distilling step-by-step은 작게는 12.5%의 training set만으로도 달성하였다.

### Distilling step-by-step outperforms standard distillation with much less unlabeled examples

이어서, unlabeled data만 available할 때, standard task distillation과 비교한 결과이다.

<img width="984" alt="9" src="https://github.com/augustinLib/All-of-NLP/assets/74291999/e2fba9e4-43d6-4efd-aca5-c84dac4ac139">

Fine-tuning과 비교했을때와 비슷하게, training example의 수가 증가할수록 꾸준히 성능 증가가 이루어지는 것을 확인할 수 있다. 마찬가지로 모든 경우에서 distilling step-by-step이 standard task distillation보다 더 좋은 성능을 내는 것도 확인 가능하며, 작게는 12.5%의 training set만으로도 full training set을 사용한 standard task distillation의 성능을 앞지른 것을 확인할 수 있다.

## Reducing model size

이어서 저자들은 model size 측면에서 기존 방법론들과 distilling step-by-step 방법론을 비교한다.

이때, training set size는 full set을 사용하는 것으로 고정하며, 220M의 T5-Base, 770M의 T5-Large, 11B의 T5-XXL을 사용한다.

LLM에 대해서도 2가지의 baseline을 설정하여 비교하였는데, Few-shot CoT와 PINTO tuning이다. Few-shot CoT는 말 그대로 Few-shot CoT를 적용한 LLM의 결과를 말하며, PINTO tuning은 앞서 말했던 rationale을 additional input으로 사용한 방법론이다. 결과적으로, model size를 다르게 바꿔가면서 standard fine-tuning, standard task distillation, distilling step-by-step, LLM Few-shot CoT, LLM PINTO tuning 결과를 비교하게 된다

이에 대한 실험 결과를 나타낸 figure를 보면서, 하나씩 결과를 살펴보도록 하겠다

<img width="984" alt="10" src="https://github.com/augustinLib/All-of-NLP/assets/74291999/ba80772d-b344-477a-9e9b-096b5a1c7b82">

### Distilling step-by-step improves over standard baselines across varying model sizes used.

우선, distilling step-by-step 방법론은 model size가 증가할수록 성능도 같이 증가하는 것을 확인할 수 있다. 그리고 모든 경우에서 standard fine-tuning과 standard task distillation의 성능을 능가함을 확인할 수 있다.

### Distilling step-by-step outperforms LLMs by using much smaller task-specific models.

우선, figure 6은 human-labeled example이 available한 경우이며, 이 경우 모든 dataset에서 LLM(540B)보다 훨씬 작은 크기의 model로도 LLM baseline의 성능을 능가할 수 있음을 보였다.

Figure 7은 unlabeled data만 available할 때인데, 이 경우엔 SVAMP를 제외한 dataset에서 LLM(540B)보다 훨씬 작은 크기의 model로도 LLM baseline의 성능을 능가할 수 있었다. SVAMP에서는 distilled model의 성능이 전반적으로 좋지 않았는데, 저자들은 이에 대한 원인으로 SVAMP dataset의 적은 data개수를 지목하였다.

### Unlabeled data augmentation further improves Distilling step-by-step

앞서 이야기했듯이, 저자들은 SVAMP dataset의 적은 data개수가 해당 dataset에서의 성능 저하의 원인이라고 추측하였다. 그래서, 이를 보완해보기 위해 SVAMP dataset과 비슷한 ASDiv dataset을 추가적으로 training dataset에 포함시켰다.

이렇게 ASDiv dataset을 추가하여 학습한 결과, 11B T5 model에서 standard distillation과 distilling step-by-step 모두 성능 향상이 있음을 확인할 수 있다.

다만, 이렇게 training data의 개수를 늘렸음에도 불구하고, 여전히 standard distillation은 LLM few-shot CoT보다 성능이 낮다. 반면에 distilling step-by-step의 경우 LLM few-shot CoT보다 3% 낮은 성능을 보였으며, 저자들은 distilling step-by-step가 50배가량 작은 model을 사용하는 효율성에도 불구하고 비슷한 성능을 냈음을 강조한다.

## Outperforming LLMs using minimum model size and least training data

저자들은 LLM 성능을 기준점으로 잡은 뒤, 해당 성능을 능가하는 distilling step-by-step과 standard fine-tuning, standard task distillation 방법론의 효율적인 training example의 개수와 model size를 찾고자 하였다. 이 또한 human-labeled setting과 unlabeled setting을 나눠서 진행하였으며, 결과는 아래와 같다

<img width="966" alt="11" src="https://github.com/augustinLib/All-of-NLP/assets/74291999/adff2ba1-2fad-4914-a869-7f62a7fecb51">

### Distilling step-by-step outperforms LLMs with much smaller models by using less data.

위의 figure(figure 8, human-labeled setting)부터 살펴보자. 우선, distilling step-by-step 방법론이 PaLM few-shot CoT보다 훨씬 작은 model을 사용하고, 전체 training set을 다 사용하지 않았음에도 더 좋은 성능을 내는 것을 확인할 수 있다. 특히, e-SNLI에서는 220M T5와 전체 training set의 0.1%만 사용했음에도 훨씬 좋은 성능을 낸 것을 확인할 수 있다.

아래의 figure(figure 9, unlabeled setting)에서도, LLM대비 더 작은 model과 전체 training set의 일부분만 사용했음에도 PaLM few-shot CoT 성능보다 더 좋은 것을 확인할 수 있다

### Standard finetuning and distillation require more data and larger model.

이에 반해, standard fine-tuning, standard task distillation 방법론의 경우 distilling step-by-step 방법론보다 전반적으로 성능이 낮은 것을 확인할 수 있다. 게다가, distilling step-by-step에 비해 특정 성능에 진입하기 위해 필요한 training example의 개수가 전반적으로 많이 요구됨을 확인할 수 있다

## Further ablation studies

지금까지는 distilling step-by-step이 model size와 required training example 측면에서 얼마나 효율적인지를 보였다. 지금부터는, distilling step-by-step의 다른 요소들이 성능에 어떠한 영향을 미치는지를 보이기 위한 비교 실험을 진행한다.

특히, 아래의 항목들에 대해 집중적으로 다룬다

- Rationale이 추출되는 LLM이 달라짐에 따라 distilling step-by-step의 효율성은 어떻게 달라지는가?
- Multi-task training 방법론과 다른 training 방법론에 대한 비교

저자들은 이와 같은 항목들을 다루기 위해 비교 실험을 진행하였고, 이때 small task-specific model은 220M T5 model로 두었으며, 전체 training set을 사용하여 학습을 진행하였다고 한다

### Distilling step-by-step works with different sizes of decently trained LLMs.

기존 실험들에서는 LLM으로 540B size PaLM model을 사용했다고 한다. 여기에 더해, 20B size의 GPT-NeoX을 추가적으로 사용해보고 결과를 비교하였다.

<img width="472" alt="12" src="https://github.com/augustinLib/All-of-NLP/assets/74291999/2ce6fe74-378e-412e-bff9-f3a1b9c8da6a">

결과를 보면, distilling step-by-step은 LLM의 size가 작아졌음에도 불구하고 여전히 standard fine-tuning에 비해 좋은 성능을 냈음을 확인할 수 있다. 그러나, LLM의 size가 작아짐에 따라 성능 하락이 존재하는것 또한 확인할 수 있다. 저자들은 이러한 현상에 대해 더 큰 size의 LLM이 higher-quality rationale을 생성하기에, 결과적으로 task를 학습함에 있어서도 성능 차이가 발생하는 것이라고 말한다

### Multi-task training is much more effective than single-task rationale and label joint prediction

LLM의 rationale을 output supervision으로 사용하여 model을 학습시킬 수 있는 방법은 다양하다. 직접적인 방법으로는, rationale $\hat{r}_i$과 label $\hat{y}_i$를 single sequence $[\hat{r}_i, \hat{y}_i]$로 concat하고, 해당 sequence를 model로 하여금 생성하게 하는 것이다. 즉, 아래와 같이 loss function이 정의된다

<img width="467" alt="13" src="https://github.com/augustinLib/All-of-NLP/assets/74291999/9bce85aa-b343-49d2-91b4-de35f53da3dc">

(논문에서는 이러한 방법론에 대해 single-task training이라고 명명한다. distilling step-by-step의 경우 multi-task training이다.)

저자들은 이러한 방법론과 standard fine-tuning, distilling step-by-step의 성능을 비교해보았다

<img width="467" alt="14" src="https://github.com/augustinLib/All-of-NLP/assets/74291999/c8ecdc52-d652-4986-b16f-99b75c33c14e">

결과를 보면, single-task training이 오히려 standard fine-tuning보다 성능이 하락하는 경우가 발생하는 것을 확인할 수 있다. 저자들은 이에 대해, rationale prediction과 label prediction을 단순히 하나의 joint task로 취급하는 것이 model의 label prediction 성능을 저해시킬 수 있다고 말한다. 결과적으로, 저자들은 본인들의 multi-task training 방법론의 우수성을 주장한다.