---
layout: post
title: "Distillation: Distilling the Knowledge in a Neural Network"
categories: [paper-review]
tags: [CV, Distillation]
paper:
    venue: "NIPS 2014 Deep Learning Workshop"
    date: "March 2015 (arXiv)"
    title: "Distilling the Knowledge in a Neural Network"
    authors: "Geoffrey Hinton, Oriol Vinyals, Jeff Dean"
    affiliation: "Google"
    paper_url: "https://arxiv.org/abs/1503.02531"
    project_url: ""
    github_url: ""
---
# Distillation: Distilling the Knowledge in a Neural Network

## Abstract

- 기존에는 Ansemble을 사용했음 그러나 이 방법은 cumbersome하고 계산량이 많다.
- 저자들은 Distilling을 통해 해당 방법을 해결할 수 있었음.
- 이는 rapidly and parallel하다.

## Intro

- 학습용 모델은 크고 느려도 되지만, 배포용 모델은 작고 빨라야한다. → 큰 모델이 학습한 “일반화 방식”을 작은 모델에게 전달하자.
- 큰 모델이 출력하는 class probability를 soft target으로 사용하는것이다.
- soft target은 부드러운 분포, hard target은 raw data 분포를 의미한다(one-hot).
- soft tartget은 entropy가 높을 수록 더 많은 정보를 담고 있으며, gradient variance도 작아서 작은 모델을 더 적은 데이터와 더 큰 learning rate로도 안정적으로 학습 시킬 수 있다.

## Distillation

- 신경망은 일반적으로 각 클래스에 대해 계산된 logit $z_i$ 에 대한 확률을 계산하기 위해 output layer인 softmax를 사용한다.

$$
q_i = \frac{\exp(z_i/T)}{\sum_j\exp(z_i/T)}
$$

- 여기서 T는 temperature를 의미하고 일반적으로 1로 사용된다.
- T가 높을 수록 확률분포는 상대적으로 softer해진다.

- 가장 단순한 형태의 distilalation에서는, transfer set 에 대해 큰 모델의 softmax를 통해 soft target distribution을 사용하여 작은 모델을 학습시킴으로써 지식을 이전한다.
- 이때 distilled model을 학습할때도 동일한 높은 temperature을 사용하지만, 학습이 끝난 뒤에는 temperature를 1로 설정하여 사용한다.

- transfer set이 일부 혹은 전부 정답 label이 알려져있다면, 이 방법은 더 향상될 수 있다.
- 정답 label을 이용해 soft target을 수정할 수 있지만, 저자들은 두 object function의 가중 평균을 통해서 더 효과적 수행할 수 있음을 발견하였다.

- 첫번째 object function은 soft target에 대한 cross entropy이며, 이는 큰 모델로부터 soft target을 생성할 때 사용한 것과 동일한 높은 temperature를 distilled model의 softmax에 적용해 계산된다.
- 두번째 objective function은 correct label에 대한 cross-entropy로, distilled model의 동일한 logit을 사용하되 temperature는 1로 설정하여 계산된다.
- 일반적으로 두번째 objective function에 대한 가중치를 낮게 부여할때 더 좋은 성능을 얻을 수 있다.
- 또한, soft target으로부터 발생하는 gradient의 크기는 1/T^2에 비례하므로, hard target과 soft target을 동시에 사용할 경우에는 soft target에서 발생하는 gradient에 T^2을 곱해주는것이 중요하다. 이렇게 하면 distillation에 사용하는 temperature를 변경하더라도, hard target과 soft target이 loss에 기여하는 상대적인 비중은 유지된다.

### Matching logits is a special case of distillation

1. gradient 관점에서 본 distillation

$$
\frac{\partial C}{\partial z_i} = \frac{1}{T}(q_i-p_i)
$$

- $T$ 가 매우 크면
    
    $$
    e^{z_i/T}\approx 1 + z_i/T
    $$
    
    - soft max 선형화
- zero mean 가정
    
    $$
    \sum_jz_j = \sum_jv_j=0
    $$
    
- 결과
    
    $$
    \frac{\partial C}{\partial z_i}\approx\frac{1}{NT^2}(z_i-v_i)
    $$
    

→결론적으로 이 문단이 하고 싶은 말은, backpropagation을 진행시키기 위해서 미분을 진행했더니 hightemperature일때, 마치 MSE와 같은 꼴이 됐다.

2. Temperature를 낮출때

- 매우 negative한 logit
    - teacher 학습 시 거의 신호 없음
    - noise일 가능성 큼
- 낮은 T 효과
    - 확률이 거의 0인 클래스 무시
    - 중요한 클래스만 집중
- Intermidate Temperature가 좋은가?
    - T 너무 큼 → noisy
    - T 너무 작음 → 정보 손실

# Conclusion

“배포용으로는 작은 모델이 필요하지만,

큰 모델이나 ensemble이 학습한 ‘일반화 방식’은 버리고 싶지 않다.

distillation은 soft target을 통해 그 일반화 지식을 작은 모델로 옮기는 방법이며,

hard label loss는 약하게 섞는 것이 가장 효과적이다.”