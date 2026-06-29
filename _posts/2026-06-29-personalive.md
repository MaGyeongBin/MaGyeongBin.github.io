---
layout: post
title: "PersonaLive: Expressive Portrait Image Animation for Live Streaming"
categories: [paper-review]
tags: [Portrait Animation, Diffusion Model, Live Streaming, Real-time Generation, Video Generation, Motion Control]

paper:
    venue: "CVPR 2026"
    date: "2025"
    title: "PersonaLive! Expressive Portrait Image Animation for Live Streaming"
    authors: "Zhiyuan Li, Chi-Man Pun, Chen Fang, Jue Wang, Xiaodong Cun"
    affiliation: "University of Macau, Dzine.ai, GVC Lab Great Bay University"
    paper_url: "https://arxiv.org/abs/2512.11253"
    project_url: "https://github.com/GVCLab/PersonaLive"
    github_url: "https://github.com/GVCLab/PersonaLive"
    arxiv_url: "https://arxiv.org/abs/2512.11253"
---

![image.png](/assets/img/2026-06-29-personalive/image.png)

## 1. Introduction

인플루언서의 라이브 스트리밍은 숏폼 비디오 소셜 미디어에서 가장 핫한 분야이다.초기 3D 아바타 방식들은 풍부한 움직임을 reenact하지 못하고, 고가의 모션 캡쳐 디바이스에 의존.반면,  portrait 애니메이션 알고리즘은 정적인 초상 이미지를 driving video에서 얻은 움직임, 즉 세밀한 표정과 자세에 따라 애니메이션화 할 수 있어 큰 가능성을 보여준다.

최근 diffusion 기반 portrait 애니메이션 방법들은 강력한 생성 능력 덕분에 주요한 패러다임으로 떠올랐다.그러나 1) 높은 계산 비용, 2) chunk-wise 처리 방식의 한계가 있다.

저자들은 portrait animation이 주로 매우 유사한 연속 프레임들 사이의 motion change를 모델링하는 작업이라고 본다. 따라서 반드시 많은 denoising step이 필요하지 않을 수 있다. 또한 독립적 chunk-wiase generation과 달리, 이전에 생성된 프레임들의 intermediate latent와 context를 조건으로 사용하여 더 길고 연속적인 생성을 직접 학습할 수 있다.

이에따라 본 저자들은 diffusion 기반 새로운 프레임워크 PersonaLive를 제안. 최근 referenceNet기반 diffusion animation 방법들의 성공을 바탕으로 새로운 구성요소를 도입.

1. Hybrid Control을 통한 Motion Transfer.
    - 본 연구에서는 implict facial represnetation 과 3D implict keypoints로 구성된 hybrid motion signal을 사용하여, 세밀한 얼굴 움직임과 머리 움직임을 동시에 제어.
    - 기존 방법에서 사용된 2D landmarks나 motion frames와 비교했을 때, 3D implict keypoints는 머리 움직임을 더 유연하고 제어 가능하게 표현.
2. Fewer-Step Appearance Distillation.
    - portrait animation의 denoising과정에 appearance redundancy가 존재한다는 점을 관찰.
    - 구조적으로, 구조적 배치와 움직임은 초기 denoising step에 형성되는 반면, 이후 많은 반복 과정은 texture와 illumination 같은 appearance detail을 점진적으로 다듬는 데 비효율적으로 사용된다. 이러한 비효율성을 해결하기 위해, 우리는 pretrained diffusion model을 compact sampling schedule에 적응시키는 appearance distillation 전략을 도입.
3. Micro-chunk Streaming Video Generation
    - 앞선 전력으로 denoising 과정을 가속화한 후, 우리는 실시간 스트리밍 응용을 위해 낮은 지연 시간과 시간적 일관성을 갖는 비디오 생성을 목표로 한다.
    - 동일한 noise level을 가진 latent에 의존하는 기존 chunk-wise generation과 달리, 우리는 autoregressive micro-chunk streaming paradigm을 채택. 이 방식은 denoising window에서 micro chunk마다 점진적으로 더 높은 noise level을 부여하여 연속적인 비디오 생성을 가능하게 한다.
    - autoregressive 방식에 내제된 exposure bias를 완화하기 위해, 우리는 학습 단계와 추론 단계 사이의 불일치를 제거하는 Sliding Training Strategy를 설계. 또한 streaming generation 중 오류 주넉을 효과적으로 완화하기 위해, 과거 프레임을 보조 reference로 적응적으로 선택하는 Historical Keyframe Mechanism(HKM)을 도입.
    
    정량적·정성적 실험 결과는 기존 방법들의 모델보다 7~22배 빠른 속도를 달성+SOTA 달성.
    
    따라서 본 논문의 기여는 아래와 같다.
    
    1. 실시간, 스트리밍 가능한 portrait animation을 위한 few-step diffusion 기반 프레임워크인 PersonaLive를 제안. 이 방법은 낮은 지연 시간과 안정적인 장기 품질을 달성한다.
    2. implicit facial representation 과 3D implicit keypoints를 결합한 hybrid motion signal을 설계하여, 세밀한 얼굴움직임과 머리움직임을 동시에 제어할 수 있게 한다. 또한 denoising 과정의 appearance redundancy를 제거하기 위한 fewer-step appearnce distillation 전략을 도입하여, 시각적 충실도를 손상시키지 않으면서 추론 효율성을 크게 향상시킨다.
    3. sliding training strategy와 historical keyfreame mechanism을 갖춘 autoregressive micor-chunk streaming generation 패러다임을 설계. 이를 통해 exposure bias 와 error accumulation을 효과적으로 완화하고, 안정적인 long-term generation을 가능하게 한다.

---

## 2. Related Work

### Diffusion-based Portrait Animation

Diffusion model은 강력한 생성 능력을 입증했다. LDM은 denoising 과정을 더 낮은 차원의 latent space에서 수행함으로써 효율성을 더욱 향상시켰다.

이러한 기반 위에서 여러 연구들은 사전 학습된 diffusion model을 확장하여, 명시적인 구조 조건을 활용하는 제어 가능하고 고품질의  portrait animation을 수행했다. 여기서 사용되는 구조 조건에는 face keypoint, face mesh redering, 원본 driving video.

이러한 방법들은 보통 ControlNet이나 PoseGuider를 사용. 또한 세밀한 얼굴 움직임을 모델링하기 위해 최근 연구들은 implicit facial representation을 도입. 이 전략은 복잡하고 미세한 얼굴 표정 디테일을 더 잘 보존하게 하며, 더 유연하고 현실적인 애니메이션을 가능하게 한다.

하지만 위의 방법들은 주로 시각적 품질과 motion consistency를 개선하는데 집중했으며, 추론 효율성은 상대적으로 간과했다. 본 연구에서는 효율적이고 시간적으로 일관된 portrait animation을 가능하게 하는 실시간 스트리밍 diffusion framework를 제안하여 이러한 한계를 해결.

### Long-term Portrait Animation

애니메이션 방법의 빠른 발전과 사용자 기대치의 증가로 인해, 시간적으로 일관된 장기 비디오를 생성하는 것이 매우 중요.

계산 제약 때문에 기존 diffusion 기반 방법들은 짧은 clip으로 학습되고, 더 긴 sequence를 만들기 위해 inference 단계에서 이를 확정하는 방식에 의존.

X-Portrait, X-NeMo는 chunk 경계 사이의 시간적 부드러움을 향상시키기 위해 prompt traveling 기법을 사용. Follow-your-emoji는 keyframe-guided interpolation을 통해 중간 프레임을 생성하는 coarse-to-fine progressive strategy를 설계. Sonic은 timestep축을 따라 이전 clip과 연결되는 time-aware shifted window를 통해 clip간 전역 연결을 구축.

하지만 이러한 발전에도 불구하고, 기존 접근법들은 여전히 실시간 스트리밍 생성에는 적합하지 않다. 몇몇 방법들은 긴 비디오의 chunk-wise streamting generation을 가능하게 하기 위해 motion frames를 활용하지만, 이는 추가적인 학습 부담을 피할 수 없다.

반면 본 연구에선느 스트리밍 가능하면서도 시간적으로 일관된 장기 portrait animation을 위해 autoregressive micro-chunk framework를 제안.

### Diffusion Model Acceleration

Diffusion model은 강력한 성능을 보이지만, 높은 계산 비용 때문에 실시간 응용과는 거리가 멀다.

기존 가속 전략은 크게 두 가지로 나눌 수 있다. 1) Model Quantization, 2) Sampling step reduction.

ADD, LCM, DMD, DMD2 는 distillation 기법을 적용한 방법으로 최근 많은 발전이 있음에도 불구하고, portrait animation 분야에서 주목받지 못했다. 본 논문에서는 실시간 portrait animaiton을 위해 diffusion distillation을 탐구.

---

## 3. Method

![image.png](/assets/img/2026-06-29-personalive/image%201.png)

Streaming Portrait Animation의 목표는 주어진 reference image와 driving video로부터 실시간이면서 낮은 지연 시간으로, 장기적이고 시간적으로 일관된 애니메이션 스트림을 생성하는 것.

형식적으로 주어진 reference image $I_R$, 연속적인 $S$ 개의 driving frame $\{I_D^1, I_D^2,, ..., I_D^S,\}$ 가 주어졌을 때, streaming portrait animation의 목적은 스트리밍 방식으로 애니메이션 시퀀스 $A_{\{1,2,..., S\}}$ 를 합성하는것. 여기서 각 프레임은 $I_R$ 로부터 얻은 외형 정보와 $\{I_D^1, I_D^2, ...,I_D^S\}$ 로부터 추출한 motion cue를 결합하여 실시간으로 렌더링. 이는 다음과 같이 표현됨.

$$
\mathcal{A}_i = \mathcal{D}(\mathcal{M}(I^i_D), \mathcal{R}(I_R)),\space\  i=1, 2, ..., S
$$

여기서 $\mathcal{D}$ 는 denoising backbone, $\mathcal{M}$ 은 motion extractor, $\mathcal{R}$ 은 appearance extrator를 의미.

먼저 hybrid motion control을 이용해 표현력 있고 견고한 motion transfer를 수행. 그 후, 중복적인 appearance refinement 과정을 압축하기 위해 fewer-step appearance distillation strategy를 도입. 마지막으로, 낮은 지연 시간과 안정적인 장기 생성을 보장하기 위해 slidign training strategy와 historical keyframe mechanism을 갖춘 micro-chunk streaming generation paradigm을 제안.

### 3.1 Image-level Hybrid Motion Training

그림과 같이 저자들은 사전학습된  diffuision model $\mathcal{D}$ 를 denoising backbone으로 사용하고, reference network $\mathcal{R}$ 을 appearance conditioning에 사용. 표현력 있고 견고한 motion control을 위해, implicit facial representation 과 3D implicit keypoints 로 구성된 hybrid conditioning signal을 사용.

구체적으로, 먼저 driving image에서 얼굴 영역을 crop 한 뒤, face motion extractor $E_f$를 사용하여 이를 1차원 facial motion embedding으로 인코딩.

$$
m_f = E_f(I_D)
$$

이렇게 얻은 $m_f$는 cross-attention layer를 통해 diffusion model $D$ 에 주입.

하지만 implicit facial representation은 주로 국소적인 얼굴 움직임에만 집중하므로, 저자들은 추가로 3D implicit keypoints를 도입하여 전역적인 자세, 위치, 크기 정보를 포착. 이를 위해, off-the-shelf 방법 $E_k$을 사용하여 driving image $I_D$와 source image $I_R$ 로부터 3D 파라미터를 추출.

$$
\begin{cases}k_{c,d},\ R_d,\ t_d,\ s_d = \mathcal{E}_k(I_D), \\k_{c,s},\ R_s,\ t_s,\ s_s = \mathcal{E}_k(I_R).\end{cases}
$$

- $k_c$: canonical keypoints, $R$: rotation, $t$: translation, $s$: scale

이후 driving 3D implicit keypoints $k_d$는 다음과 같이 변환된다.

$$
k_d = s_d\cdot k_{c,s} R_d+t_d
$$

마지막으로, 추출된 3D implicit keypoints $k_d$ 는 pixel space로 매핑된 뒤, PoseGuider를 통해 diffusion model $D$ 에 주입.

즉, source image의 외형은 유지하고, driving image에서 뽑은 표정 정보와 3D 머리 움직임 정보를 같이 써서 더 자연스럽게 얼굴 애니메이션을 만드는 방식.

### Fewer-step Appearance Distillation

Hybird motion control을 기반으로, 저자는 portrait animation에서 각 프레임의 motion과  structural layout이 대부분 가장 초기의 denoising step에서 결정되는 반면, 이후 반복 과정은 주로 appearance detail을 정제한다는 점을 관찰. 이러한 관찰은 denoising 과정에 상당한 중복성이 존재함을 보여줌.

위의 동기를 바탕으로, 저자는 중복적인 refinement 과정을 compact sampling scheule $\{t_i\}_{i=1}^N$ 로 압축하기 위한 fewer-step appearance distillation 전략을 도입. 구체적으로 $z_{noise}\sim\mathcal{N}(0,I)$ 에서 시작하여, denoising step $n\in [1,N]$ 을 무작위로 샘플링하고 n번의 denoising iteration을 수행하여 intermediate noise-free state $\hat{z_0}$ 를 얻는다. 이후 이는 pixel space로 디코딩 되어 $\hat x = V_d(\hat z_0)$.

여기서 예측 이미지 $\hat x$ 는 MSE loss, LPIPS loss, adversarial loss 를 결합한 hybrid objective를 사용하여 해당 ground-truth frame $x^{gt}$ 로부터 supervision을 받는다.

$$
\mathcal{L}_{distill} = \mathcal{L}_2(\hat x, x^{gt})+ \lambda_{lpips}\mathcal{L}_{lpips}(\hat x, x^{gt}) + \lambda_{adv}\mathcal{L}_{adv}(\hat x)
$$

여기서 $\lambda_{lpips}$와 $\mathcal{L}_{lpips}$ 는 balancing coefficient. 전체 diffusion process를 통해 backpropagation을 수행하면 과도한 메모리 소비가 발생. 계산 효율성을 높이기 위해, 저자들은 마지막 denoising step에 대해서만 gradient를 전파하며, stochastic step sampling을 통해 학습 전반에 걸쳐 모든 middle timestep이 supervision을 받을 수 있도록 한다.

### 3.3 Micro-chunk Streaming Video Generation

이미지 애니메이션 모델을 비디오 생성으로 확장하기 위해, 저자는 denoising backbone $D$ 에 temporal module을 통합. 그러나 기존 방법처럼 하나의 denoising window 안에 있는 모든 프레임에 동일한 noise level을 부여하는 대신, denoising window를 점진적으로 더 높은 noise level을 갖는 여러 개의 micro-chunk로 나눈다.

형식적으로, step $s$ 에서의 denoising window는 $N$ 개의 micro-chunk 집합으로 정의.

$$
W_s = \{C_s^1, C_s^2, \dots, C_s^N\}
$$

$$
C_s^n = \{z_{s,i}^{t_n} \mid i = 1, 2, \dots, M\}, \quad t_1 < t_2 < \dots < t_N
$$

- $C_s^n$ 은 $M$ 개의 프레임으로 구성된 $n$ 번째 micro-chunk를 의미.

각 denoising step 이후, 모든 chunk는 더 낮은 noise level로 이동하며, 첫번째 chunk는 방출 가능한 $M$ 개의 clean frame을 생성. 이후 denoising window는 한 chunk만큼 앞으로 이동하고, 마지막에는 Gaussain noise로 초기화 된 새로운 noisy chunk $C_{noise} = \{\epsilon_i\}_{i=1}^M$ 추가.

이러한 streaming processing paradigm은 overlapping region 없이 연속적인 프레임 생성을 가능하게 하며, 시간적 일관성과 낮은 지연 시간을 모두 보장. 그러나 효율적임에도 불구하고, streaming generation 은 긴 비디오 시퀀스를 생성할 때 여전히 exposure bias와 error accumulation 문제를 겪는다. 이를 해결하기 위해, 저자들은 장기 생성을 안정화하고 시간적 일관성을 높이기 위한 sliding training strategy와 historical keyframe mechanism을 설계.

**Sliding Traning Strategy**

Streaming generation에서 exposure bias는 주로 학습과 추론 사이의 불일치에서 발생한다. 학습 중에는 모델이 ground-truth frame에서 유도된 입력을 통해 학습된다. 그러나 추론 중에는 모델이 스스로 생성한 예측 결과에 의존해야하며, 이 예측 결과는 필연적으로 ground-truth data의 분포에서 벗어나게 된다. 이로 인해 시간적 오류가 누적.

이 문제를 완화하기 위해, 저자는 학습 중에 streaming generation 과정을 시뮬레이션 하여 모델이 자신의 예측 오류를 직접 경험하고 학습하도록 한다.

$$
C_0^n =\left\{\sqrt{\bar{\alpha}_{t_n}} z_i^{gt}+\sqrt{1-\bar{\alpha}_{t_n}} \epsilon_i\right\}_{i=1}^{M}
$$

- $\epsilon_i \sim \mathcal{N}(0, I), \alpha_{t_n}$ 은 noise scheduling parameter 이고, $\bar\alpha_{t_n} = \prod_{i=1}^{t_n}\alpha_i$

마지막 chunk $C_0^N$ 은 random noisy chunk $C_{noise}$ 로 초기화. 각 denoising step 이후, denoising window는 한 chunk 만큼 앞으로 이동하고, 마지막에는 새로운 noisy chunk가 추가된다.

계산 비용을 줄이기 위해, 저자는 일부 denoising window에 대해서만 gradient를 계산하고, 단일 denoising step을 통해서만 gradient를 전파. 전체 학습 objective는 apperance distillation 단계와 동일.

![image.png](/assets/img/2026-06-29-personalive/image%202.png)

위의 그림에서 보이듯 implict motion signal을 interpolation 하면 source motion driving motion으로 부드럽게 전환할 수 있다. 이러한 특성을 활용하여, 저자는 Motion-Interpolated Initialization 전략을 도입. 이 전략은 reference image $I_R$ 와 보간된 implict motion signal을 결합하여 첫 번째 denoising window를 구성함으로써, 추론 과정을 학습 설정과 맞춘다.

**Historical Keyframe Mechanism**

Reference image에 의해 명시적으로 제약되지 않는 영역, 예를 들어 가려진 영역을 합성할 때, diffusion sampling에 내재된 확률성은 프레임 간 미묘한 apperance variation을 발생시킬 수 있다.

Streaming generation 환경에서는 이러한 불일치가 점차 누적되어, 시간이 지남에 따라 temporal drift와 시각적 안정성 저하를 초래할 수 있다. 이를 완화하기 위해, 우리는 이전에 생성된 결과 중 대표적인 프레임인 historical keyframe을 auxiliary reference로 도입한다. 이를 통해 모델은 안정적인 과거 단서를 활용하여 장기 streaming synthesis 동안 appearance consistency를 유지할 수 있다.

저자는 history bank $B_{his}$ 와 motion bank $B_{mot}$ 를 유지한다. History bank는 historicla keyframe에서 추출된 reference feature $\{h_0, h_1, ..., \}$ 를 저장하고, motion bank는 이에 대응하는 motion embedding $\{m_0, m_1, ...\}$ 를 저장.

각 denoising step 이후, 첫 번째 프레임의 현재 motion embedding $m_f$가 주어지면, 우리는 이를 $B_{mot}$ 와 비교하여 다음과 같이 유사도를 측정.

$$
d = \min_{i=0,1,\dots} \left\| m_f - m_i \right\|_2
$$

여기서 $d$ 가 기준치를 넘는다면, 현재 프레임은 keyframe으로 식별. 이후 해당 프레임은 reference feature, motion embedding 는 각각 $B_{his}, B_{mot}$ 에 추가.

이후 추론 과정에서는 선택된 historical feature들이 source image feature $h_0$ 와 concat 되고, spatial module을 통해 diffusion backbone에 주입되어 temporal consistency를 향상시킨다.

---

## 4. Experiments

### 4.1 Evaluation and Comparisons

![image.png](/assets/img/2026-06-29-personalive/image%203.png)

![image.png](/assets/img/2026-06-29-personalive/image%204.png)

**Baselines**

- GAN 기반: LivePortrait
- Diffusion 기반: X-Portrait, Follow-your-Emoji, Megactor-Σ, X-NeMo, HunyuanPortrait

**Datasets**

Self-reenactment: TalkingHead-1KH
Cross-reenactment: 직접 구축한 LV100 (100개 portrait + 100개 long video, 대부분 1분 이상)

**Results**

Self-reenactment에서 대부분 지표 best/2nd
Cross-reenactment에서 FVD, tLP 기준 최고 temporal coherence
효율성: 15.82 FPS, latency 0.253s — 기존 diffusion 대비 7~22× 빠름

### 4.2 Ablation Studies

**Appearance Distillation**

![image.png](/assets/img/2026-06-29-personalive/image%205.png)

**Micro-chunk Streaming ablation**

![image.png](/assets/img/2026-06-29-personalive/image%206.png)

**Failure case**

![image.png](/assets/img/2026-06-29-personalive/image%207.png)

---

## 5. Conclusion

1. Hybrid Control 기반 image animation framework
2. Appearance Distillation + Micro-chunk Streaming으로 real-time/low-latency 달성
3. Sliding Training Strategy + HKM으로 exposure bias/error accumulation 해결

### Limitations

1. temporal redundancy 미활용: 연속 프레임간의 중복 정보를 명시적으로 활용 하지 않음
2. Out-of-domain 일반화 부족: 학습이 human facial data 위주