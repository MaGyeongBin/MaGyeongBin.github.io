---
layout: post
title: "Multimodal Machine Learning: A Survey and Taxonomy"
categories: [paper-review]
tags: [Multimodal, Survey, Taxonomy]

paper:
    venue: "IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)"
    date: "2019"
    title: "Multimodal Machine Learning: A Survey and Taxonomy"
    authors: "Tadas Baltrušaitis, Chaitanya Ahuja, Louis-Philippe Morency"
    affiliation: "Carnegie Mellon University"
    paper_url: "https://ieeexplore.ieee.org/document/8269802"
    project_url: ""
    github_url: ""
---
앞서, 본 논문 리뷰는 간략히 진행됐습니다. 논문에 대한 출처는 상단 배너에 작성해놨으니 자세한 내용이 궁금하시면 직접 논문을 읽으시거나, 저에게 메일을 보내주시면 좋겠습니다. 

## Introduction

- 우리 주변 세계는 multiple modalities로 구성되어 있다. 우리는 vision, audio, touch, smell를 느낄 수 있다. modality는 어떤 것이 발생하거나 경험되는 방식을 의미.
- 어떤 연구 문제나 데이터셋이 여러 modality를 포함할 경우 이를 multimodal이라고 한다. 본 논문은 주로 다음 세 가지 modality에 초점을 둔다.
    - Natural language: written 또는 spoken 형태
    - Visual signals: images 또는 videos
    - Vocal signals: sounds 및 prosody, vocal expressions와 같은 para-verbal 정보
- Artificial Intelligence가 세계를 이해하려면 multimodal messages를 해석하고 추론할 수 있어야 한다. Multimodal Machine Learning은 여러 modality로부터 정보를 처리하고 상호 연관시키는 모델을 구축하는 것을 목표로 한다.
- 초기의 audio-visual speech recognition 연구부터 최근의 language-vision models에 이르기까지, multimodal machine learning은 빠르게 성장하는 multidisciplinary field이며 큰 잠재력을 가진다.
- 그러나 multimodal data는 heterogeneity를 가지기 때문에 고유한 computational challenges를 야기한다. 여러 modality에서 학습하면 modality 간 correspondence를 포착하고 자연 현상에 대한 더 깊은 이해를 얻을 수 있다. 본 논문은 multimodal machine learning에서 핵심이 되는 다섯 가지 technical challenges를 정의하고 탐구한다.
- 이 taxonomy는 단순한 early fusion / late fusion 구분을 넘어 다음 다섯 가지로 구성된다.
- **1) Representation**
    - 첫 번째 핵심 과제는 multimodal data를 어떻게 표현하고 요약할 것인가이다.
    - 목표는 여러 modality의 complementarity와 redundancy를 활용하는 representation을 학습하는 것이다. 그러나 multimodal data의 heterogeneity 때문에 이를 구성하는 것은 어렵다.
- **2) Translation**
    - 두 번째 과제는 한 modality의 데이터를 다른 modality로 translate (map) 하는 문제이다.
    - 문제는 단순히 데이터 형식이 다르다는 점뿐 아니라, modality 간 관계가 open-ended하거나 subjective하다는 점이다.
- **3) Alignment**
    - 세 번째 과제는 서로 다른 modality의 sub_element 간 직접적인 관계를 찾는 것이다.
    - Ex) 요리 recipe의 단계와 요리 video의 장면을 align하는 문제, 이를 해결하기 위해서는
        - 서로 다른 modality 간 similarity 측정
        - long-range dependencies 처리
        - ambiguity 해결
- **4) Fusion**
    - 네 번째 과제는 두 개 이상의 modality 정보를 결합하여 prediction을 수행하는 것이다.
    - Audio-visual speech recognition에서 lip motion (visual)과 speech signal (audio)을 결합하여 단어 예측
- **5) Co-learning**
    - 다섯 번째 과제는 modality 간 knowledge transfer이다.
    - 이는 다음과 같은 방법에서 나타난다:
        - co-training
        - conceptual grounding
        - zero-shot learning
    
    → Co-learning은 한 modality에서 학습된 knowledge가 다른 modality 모델 학습에 어떻게 도움을 줄 수 있는지를 탐구한다. 특히 annotated data가 부족한 modality에서 매우 중요하다.
    

## Multimodal Representation

- Multimodal representation은 여러 modality의 정보를 함께 사용하는 표현이다.
- 여러 modality를 표현하는 것은 다음과 같은 어려움을 포함한다:
    - 서로 다른 heterogeneous source를 어떻게 결합할 것인가?
    - 서로 다른 noise 수준을 어떻게 처리할 것인가?
    - missing data를 어떻게 다룰 것인가?
- 최근(2019) speech recognition및 visual object classification 의 성능 도약은 representation의 중요성을 잘 보여준다.
- Bengio et al. 은 좋은 representation이 가져야 할 특성으로 다음을 제시
    - Smoothness
    - Temporal coherence
    - Spatial coherence
    - Sparsity
    - Natural clustering
- Srivastava & Salakhutdinov 은 multimodal representation에 대해 추가적인 바람직한 특성을 제시
    1. Representation space에서의 similarity는 개념 간 similarity를 반영해야 한다.
    2. 일부 modality가 없더라도 representation을 얻을 수 있어야 한다.
    3. 관측된 modality로부터 missing modality를 복원할 수 있어야 한다.
- Unimodal representation은 오랫동안 연구되어 왔다.
    - Hand-designed feature → Data-driven feature
- 본 논문은 multimodal representation을 두 가지로 구분한다:
    - 1. Joint Representation
        - 단일 함수로 multimodal 표현 생성
        - 여러 unimodal signal을 하나의 동일한 representation space로 결합
        - $\mathbf{x}_m = f(\mathbf{x}_1, \ldots, \mathbf{x}_n)$
        
        → 즉, 여러 modality를 함께 입력받아 하나의 representation으로 변환한다.
        
    - 2. Coordinated Representation
        - 각 modality를 별도로 처리
        - 하지만 representation space에서 similarity constraint를 부여하여 coordinated space로 매핑
        - $(f(\mathbf{x}_1) \sim g(\mathbf{x}_2))$
            - 각 modality는 독립적인 projection function을 가짐 (f, g)
            - multimodal space로의 투영은 독립적
            - 그러나 결과 공간은 서로 coordinated (유사성 정렬) 됨

### Joint Representations

- Joint representation은 여러 unimodal representation을 하나의 multimodal space로 함께 투영하는 방식
- 이 방식은 주로training과 inference 단계 모두에서 multimodal 데이터가 존재하는 경우에 사용된다.
- 가장 단순한 예는 각 modality feature를 단순히 concatenation하는 방식(early fusion).
- 이 섹션에서는 단순 결합을 넘어 보다 발전된 joint representation 방법을 다룬다:
    - Neural Networks
    - Probabilistic Graphical Models
    - Recurrent Neural Networks
- **Neural Network**
    - Neural network는 unimodal representation에서 매우 성공적인 방법이다
        1. 각 modality는 여러 개의 개별 neural layer를 통과한다.
        2. 이후 하나의 hidden layer에서 modality들을 joint space로 projection한다.
        3. 생성된 joint representation은 end-to-end training이 가능하다.
        
        → 즉, representation 학습과 task 학습이 동시에 이루어진다.
        
        → 이로 인해 multimodal representation learning과 multimodal fusion은 밀접하게 연결된다.
        
    - Neural network는 많은 labeled data를 요구한다. 따라서 Unsupervised pre-training, Related domain에서 supervised pre-training 같은 방식들을 사용.
        - Stacked denoising autoencoder를 각 modality에 적용 → multimodal autoencoder layer로 결합
        - Multimodal autoencoder + reconstruction loss
- Probabilistic Graphical Model 기반 Representation
    - Latent random variable을 활용하여 representation을 구성할 수 있다 [19].
    - Deep Boltzmann Machines (DBM)
        - Restricted Boltzmann Machine (RBM)을 여러 층으로 쌓은 구조
        - 각 층은 점점 더 추상적인 representation을 학습
        - 장점
            - Generative model
            - Supervised data 없이 학습 가능 (unsupervised)
            - Missing modality를 자연스럽게 처리 가능
            - 한 modality가 주어졌을 때 다른 modality 생성 가능
        - 단점
            - Training이 매우 어려움
            - High computational cost
            - Approximate variational training 필요
- Sequential Representation (RNN 기반)
    - RNN의 hidden state는 시점 t까지의 시퀀스를 요약한 representation으로 볼 수 있다.
    - 특히 encoder-decoder framework에서 1)Encoder: 시퀀스를 hidden state에 압축 2)Decoder: 해당 representation으로부터 복원
- Joint representation은
    - 여러 modality를 하나의 space에서 통합
    - End-to-end 학습 가능
    - Task-specific optimization 가능
- 그러나:
    - Missing modality 처리에 약함
    - 학습 난이도 높음
    - Computational cost 큼

### Coordinated Representations

- Coordinated representation은 joint representation과 달리 각 modality를 독립적으로 표현하되, 특정 constraint을 통해 동일한 multimodal space에서 coordinate되도록 만드는 방식이다.
- **Similarity 기반 Coordinated Representation**
    - 서로 대응되는 modality 간 representation 거리를 줄임
    - 대응되지 않는 쌍은 거리를 크게 유지
- **Structured Coordinated Representation**
    - 단순 similarity 제약을 넘어서 공간에 structure를 강제하는 방식. 적용 분야에 따라 제약 구조가 달라짐:
    - Cross-Modal Hashing
        - 고차원 데이터를 binary code로 압축
        1. N-dimensional Hamming space
        2. 같은 객체(다른 modality)는 유사한 hash
        3. Similarity-preserving
    - Order-Embeddings
        - Multimodal space에 partial order를 도입
        - 계층적 구조 표현
        
        → 의미적 포함 관계를 공간 구조에 반영
        
    - CCA 기반 Coordinated Representation
        - 두 modality 간 correlation 최대화
        - orthogonality 유지
        - 선형 projection
    - Coordinated representation의 핵심
        - 각 modality를 독립적으로 학습
        - 동일한 공간에서 similarity / structure 제약을 통해 정렬
        - 장점:
            - Cross-modal retrieval에 매우 적합
            - Missing modality 대응 유리
            - 구조적 의미 표현 가능
        - 단점:
            - Joint representation보다 통합적 feature interaction은 약할 수 있음

## Translation

- Multimodal machine learning의 중요한 한 축은 한 modality에서 다른 modality로 **translation**.
- 주어진 한 modality의 entity를 다른 modality로 생성하는 것이 목표이다.

### **Example-based**

- Retrieval-based: 가장 유사한 사례를 그대로 적용(Ex: dictionary에서 가장 가까운 샘플 찾기)
- Combination-based: 여러 retrieved instance를  조합하여 새로운 결과 생성

### Gerantive Approaches

- Generative 접근은 unimodal source를 입력받아 다른 modality를 생성할 수 있는 모델으로 1)Source modality의 이해 2)Target modality 생성 을 수행.
- Grammar-based model
    - predefined grammar/template 사용
    - Target domain을 제한하여 생성 문제 단순화
- Encoder-Decoder models
    - Source → Encdoer → Latent vector → Decoder → Target
- Coninuous generation models
    - RNN/LSTM: Encoded vector를 initial hidden state로 사용, 이후 sequence 생성

## Alignment

- Multimodal alignment는 두개 이상의 modality에서 각 인스턴스의 하위 요소 간의 관계와 대응 관계를 찾는 과정.

### Explicit Alignment

- 두개 이상의 modality에서 각 하위 요소 간 정렬을 주요 목표로 삼는 모델
- 가장 중요한 요소는 similarity metric ← 유사도 계산을 기본 요소로

### Implicit Alignment

- 정렬 자체가 목적이 아닌, 다른 작업을 수행하기 위한 중간 latent 단계
- 다른 task 수행 중 자연스럽게 alignment 학습

## Fusion

- 기존 연구들은 early fusion, late fusion, hybrid fusion 접근을 중심으로 정리해 왔다.
- 기술적으로, multimodal fusion은 여러 modality의 정보를 통합하여 최종 예측 결과를 도출하는 과정
- Fusion의 세가지 주요 강점
    - Robustness 향상
    - Complementary 정보 활용
    - Missing Modality 대응

### Model-Agnostic Approaches

- Early Fusion
    - Feature 추출 직후 modality를 결합
    - 일반적으로 feature concat 방식
- Late Fusion
    - 각 modality가 독립적으로 예측 수행
    - 이후 decision 값을 예측
- Hybrid Fusion
    - Early + Late fusion
    - Feature 수준과 Decision 수준 모두 활용

### Model-Based Approahces

- Model-agnostic 접근은 구현이 쉽지만, 본질적으로 multmodal 데이터를 위해 설계된 방법은 아니다.
- **Kernel-based methods**
    - Multiple Kernel Learning (MKL)은 Kernel SVM의 확장으로, 각 modality에 서로 다른 kernel을 사용할 수 있게 한다. Kernel은 데이터 간 similarity function로 볼 수 있으므로, modality별 kernel을 사용하는 것은 heterogeneous 데이터 융합에 적합하다.
- **Graphical Models**
    - multimodal fusion에서 오랫동안 사용되어 왔다.
    - Generative models → joint probability 모델링
    - Discriminative models → conditional probability 모델링
- **Neural Networks**
    - 대규모 데이터 학습 가능
    - Representation + Fusion end to end 학습 가능
    - 복잡한 decision boundary 학습 가능
    - 기존 비신경망 모델보다 높은 성능

## Co-Learning

- Co-learning은 한 modality의 지식을 활용하여 다른 modality의 모델링을 돕는 것을 의미.
    - Annotated data 부족
    - Noisy input
    - Unrealiable label
- 즉 데이터가 부족한 환경에서 mulitmodal 접근의 강점을 극대화 하는 전략

### Parallel Data

- 두 modality가 동일한 인스턴스를 공유 Ex) Audio recording → 해당 video를 공유
- **Co-training**
    - Co-training은 label sample이 적을때 multimodal 구조를 활용하여 추가 labeled 데이터를 생성하는 방법
    - 각 modality에서 weak calssifier 학습
    - 서로 unlabeled 데이터에 pseudo-label 제공
    - 반복적으로 labeled set 확장
- **Transfer Learnning**
    - Multimodal representation learning 방식은 한 modality의 표현을 다른 modality 표현 개선에 활용.

### Non-Parallel Data

- 동일한 인스턴스를 공유할 필요 없다. 그러나, 공통된 category 또는 개념만 공유해야 한다.
- 더 나은 representation 학습
- 의미적 개념 이해 향상
- Unseen object recognition 수행
- **Non-parallel transfer learning**
    - Data-rich / clean modality에서 학습된 표현을
    - Data-poor noisy modality개선에 활용

### Hybrid Data

- 서로 non-parallel 한 두 modality를 하나의 공유 modality 또는 데이터셋을 연결
- A ↔ C, B ↔ C 는 존재하지만 A ↔ B 직접 대응은 없음

→이 구조를 통해 A와 B 사이의 관계를 간접적으로 학습한다.