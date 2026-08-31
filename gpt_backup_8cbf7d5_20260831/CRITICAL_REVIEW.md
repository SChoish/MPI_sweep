# GPT 통합 비평 및 AAAI-26 수정 기록

## 결론

현재 증거의 중심은 local discretization accuracy가 아니라 **finite-step bootstrap-exposure decoupling**입니다. Proximal `K=1,2,3,4`의 전체 고정-budget grid는 두 disjoint host-separated seed 쌍에서 각각 동일가중 `T>=4` aggregate의 `K=1<2<3<4` 순서를 재현했고, pooled four-seed 결과에서도 mean score와 collapse risk가 함께 개선됩니다. 이는 `K`를 실제적인 안정성 제어변수로 확립합니다. Ideal proximal/JKO 분석은 이 구조의 국소 기하를 설명하지만, live one-Adam-step chain을 보증하거나 routing·re-centering·actor compute의 개별 인과효과를 대신 식별하지는 않습니다.

이번 AAAI-26 원고는 이 경계를 명시적으로 반영했습니다. 핵심 정체성은 다음과 같습니다.

> BAR is a finite-step routed refinement procedure: depth separates critic-facing movement from deployment reach and reproducibly expands the tested high-budget stability envelope.

## 이전 GPT 비평의 정정

이전 GPT 원고는 `K`에 따라 critic initialization subkey가 달라지는 것을 중대한 confound로 보고 동일 critic initialization rerun을 필수로 제안했습니다. 이 판단은 철회합니다.

각 깊이의 randomized training procedure를 `Y_K(omega)`라 하면 complete sweep의 estimand는

```tex
\mu_K = \mathbb E_{\omega}[Y_K(\omega)]
```

입니다. 각 `K`에서 유효한 난수 표본을 사용한 seed average는 동일 initialization을 공유하지 않아도 이 marginal mean을 추정합니다. 공통 initialization/common random numbers는 방법 차이의 Monte Carlo variance를 낮출 수 있지만 validity 조건은 아닙니다. 단, paired rescue count나 sign count는 실제 seed coupling 아래의 서술적 수치로 해석해야 합니다.

또한 사용자가 말한 “K는 거의 의미없다”는 말은 **이 seed 비판이 의미 없다는 지적**이었지, refinement depth `K` 자체를 논문에서 제거하라는 뜻이 아니었습니다. 수정 원고는 `K`를 핵심 설계변수로 유지하되, `K`가 local coefficient, persistent actor count, optimizer calls, re-centering, normalization trajectory, routing을 함께 바꾸는 bundle intervention임을 명시합니다.

## P0 수정 사항

### 1. `T`의 코드상 의미

Trainer에서 `T`는 canonical scale-normalized TD3+BC의 `tau=alpha/2`입니다. 따라서 `K=1`은 `alpha=2T`인 baseline mapping과 정확히 같고, depth `K`에서는 각 hop이 `h=T/K`, critic coefficient `2h/C_k`를 사용합니다. 원고는 `T`를 nominal finite-step coefficient budget으로 정의하며 compute budget이나 realized action distance로 해석하지 않습니다.

### 2. 구현과 일치하는 `C_k`

기존 단일 `C_k` 정의는 구현을 정확히 표현하지 못했습니다. 수정 원고는 다음 네 경우를 분리합니다.

```tex
C_{1,t}^{\mathrm P}
= \operatorname{sg}\!\left(
\mathbb E_s|\widehat Q_t(s,\mu_{1,t}^{\mathrm{pre}}(s))|+\epsilon
\right),
```

```tex
C_{k,t}^{\mathrm P}
= \operatorname{sg}\!\left(
\mathbb E_s|\widehat Q_t(s,\operatorname{sg}\mu_{k-1,t}^{+}(s))|+\epsilon
\right),\quad k\ge2,
```

```tex
C_{1,t}^{\mathrm L}
= \operatorname{sg}\!\left(
\mathbb E_{(s,a)\sim\mathcal D}|\widehat Q_t(s,a)|+\epsilon
\right),
```

```tex
C_{k,t}^{\mathrm L}=C_{k,t}^{\mathrm P},\quad k\ge2.
```

따라서 linearized 경로는 action metric과 nominal coefficient는 맞지만 첫 hop의 Q-normalization point는 다릅니다. 원고는 이를 `action-metric-matched projected linearization`이라고 부르며, fully scale-matched control이라고 주장하지 않습니다.

### 3. 평가 주기

모든 headline score는 `10^6` critic-update checkpoint에서 10 episode를 평균한 D4RL normalized score입니다. Intermediate evaluation cadence는 run family마다 달랐습니다: archived configs는 K1/P2/P3에서 50,000-update cadence, L2에서 mixed 50,000/final-only, L3에서 final-only를 기록하며 exact K4 config는 보존되지 않았습니다. 따라서 원고는 하나의 cadence를 전체 sweep에 일반화하지 않고 공통 final checkpoint만 score estimand로 사용합니다.

### 4. Taylor remainder

`O(beta^3)`를 유지하는 곳에는 Hessian local Lipschitz를 가정했습니다. 단지 `C^2`만 가정하면 안전한 remainder가 `o(beta^2)`임을 본문과 부록에 명시했습니다. ReLU critic에 대해서는 activation boundary를 피한 piecewise-smooth local statement로 제한합니다.

### 5. dataset 이름

모든 표와 그림에서 `E`는 정확히 `expert-v2`를 뜻합니다. 이 실험에 쓰이지 않은 `ME`와 `medium-expert-v2` 표기는 제거했습니다. 부록에 아홉 개 dataset ID를 구성 규칙과 함께 명시했습니다.

## 가장 중요한 이론-구현 간극

이상적 local composition은

```tex
\Psi_h(x)=x+h f(x)+h^2p(x)+O(h^3)
```

인 near-identity map을 가정합니다. 실제 구현에서는 later actor가 predecessor parameter copy가 아니라 독립적으로 초기화되어 persistent하게 유지되며, 각 delayed update에서 Adam 한 번만 수행됩니다. `h=0`이어도 두 actor가 다르면 anchoring loss gradient가 남으므로 live map은 일반적으로 `Psi_0 = Id`를 만족하지 않습니다.

따라서 다음 세 층위를 분리했습니다.

1. 정확히 푼 fixed-critic proximal operator에 대한 variational statement;
2. 그 operator의 local explicit-implicit expansion과 fixed-depth composition;
3. 해당 loss를 한 번의 Adam step으로 amortize하는 실제 BAR training procedure.

1--2는 3의 자동 보증이 아닙니다. Frozen-critic audit도 ideal action fixed point를 검사할 뿐 live neural optimizer chain을 검사하지 않습니다.

## 실증적 강점

### Complete grid

`T >= 4`인 상위 7개 budget을 각 task 안에서 seed와 함께 평균하고 9개 task를 동일가중한 four-seed proximal 결과는 다음과 같습니다. 이는 pointwise monotonicity 주장이 아니라 명시된 stress-region aggregate입니다.

| `K` | High-budget mean | score<20 cells | score<20 raw runs |
|---:|---:|---:|---:|
| 1 | 25.38 | 34/63 | 142/252 |
| 2 | 41.20 | 22/63 | 111/252 |
| 3 | 51.92 | 16/63 | 84/252 |
| 4 | 63.97 | 8/63 | 54/252 |

`K=4-K=1`의 task-level high-budget mean difference는 `+38.59`, descriptive task-resampling interval은 `[21.39,54.63]`, 9개 task 중 8개에서 양수입니다. `K=4-K=3` 평균은 `+12.06`, interval `[3.84,21.11]`, positive task count는 7/9입니다. Cellwise median은 `+.34`이지만 K3 collapse 8개가 K4에서 복구되고 reverse transition은 0개입니다. 이 8개 cell이 63개 high-budget cell 전체에 걸친 K4-K3 gain 합의 `54.7%`를 차지하므로, 추가 깊이의 aggregate 평균 이득 대부분이 low-score rescue에서 온다는 해석이 직접 뒷받침됩니다.

`K=4-K=3` raw-run collapse-risk difference는 `-.119`, task-resampling interval은 `[-.218,-.024]`입니다. 최신 four-seed grid에서는 연속 score와 thresholded collapse risk가 모두 K4 방향이며, 두 disjoint host-separated seed 쌍은 각각 `K=1<2<3<4` high-budget aggregate ordering을 재현합니다.

### Target-action audit

3-hop proximal 대 TD3+BC의 sample-anchored next-state target-action displacement는 88/90 pair에서 더 작고 median ratio는 .658입니다. `(environment, seed)` 18-cluster bootstrap에서 lower-fraction interval은 `[.944,1]`, mean-difference interval은 `[-.279,-.111]`입니다. K2 대비 K3도 84/90, ratio .903으로 방향이 유지됩니다.

이 결과는 policy-to-policy Wasserstein distance를 직접 측정한 것이 아닙니다. 실제 metric은

```tex
\widehat D_{\rm target}
= \mathbb E_i\|\mu_1^-(s_i')-a_{i+1}\|_2^2/d
```

이며 paired next dataset action을 anchor로 사용합니다. 그래서 원고의 표현을 `sample-anchored next-state target-action displacement`로 고쳤습니다.

Final proxy는 current states, current dataset actions, online final actor를 사용하므로 target metric과 domain 및 Polyak status가 다릅니다. P3가 TD3보다 final proxy가 작은 경우는 36/90이고 median ratio는 1.057이지만, 몇 개의 큰 contraction 때문에 arithmetic mean difference는 음수입니다. 원고는 이를 “typical-cell contraction은 없지만 heterogeneous하다”고 기술합니다. 이 270-checkpoint mechanism audit은 TD3+BC/P2/P3만 직접 측정하므로 P3의 exposure signature를 지지하며, P4 movement를 측정한 것으로 확장하지 않습니다.

### Critic failure

Collapsed run의 TD-error p99, critic magnitude, critic loss는 stable run보다 수십 자릿수까지 커지며, common frozen critic에서 actor-chain gain의 중앙값이 stable 양수에서 collapsed 음수로 바뀝니다. 이는 cross-critic disagreement의 강한 failure signature지만 인과 식별은 아닙니다.

Simulator calibration은 state pooling 숫자 `-38.7`을 checkpoint-level statistic처럼 쓰던 문제를 고쳤습니다. Main text의 primary는 checkpoint median들의 median인 `-41.4` 대 `1.70e12`입니다. Ranking subset은 dead-zone에 민감하므로 부록의 exploratory screening으로만 남겼습니다.

### Route shadow

Primary aggregation(state median -> four-checkpoint mean -> run)에서는 final-route error delta가 8/8 양수입니다. 그러나 final checkpoint만 보면 5/8입니다. 또한 cell selection은 visible frontier를 본 뒤 이루어진 post-hoc targeted selection이고, shadow critic은 actor를 구동하지 않습니다. 따라서 본문은 두 수치를 함께 밝히고 own-continuation calibration이나 return에 대한 causal routing effect를 주장하지 않습니다.

## 재현성 판정

Compact claim audit는 통과했습니다.

- target exposure 270/270;
- failure diagnostics 180/180;
- simulator 60 checkpoints, 720 states;
- route intervention 8 runs, 32 arm/checkpoint pair checks;
- primary-key duplicate, NaN, Inf, state-pair mismatch 없음.

그러나 multi-GB raw NPZ/per-state material은 이 bundle에 복제하지 않았고, 일부 TD residual quantile은 보존된 run-level dump에서 재사용되며, historical control launcher와 exact lockfile도 불완전합니다. 그러므로 공식 31문항 checklist는 source/analysis reproducibility와 full end-to-end retraining reproducibility를 구분하고, 구체적 누락에 따라 `Partial` 또는 `No`로 답합니다.

## 남은 범위와 재현성 한계

핵심 empirical claim을 막는 P0는 이 AAAI-26 버전에 반영했습니다. 남은 범위는 다음과 같습니다.

- proximal cross-depth grid는 four seeds이지만 linearized grid와 mechanism audits는 two seeds임;
- newer proximal seed pair의 score matrices는 보존됐지만 exact per-run config manifest와 code hash는 저장소에 없음;
- task가 9개지만 dynamics family는 3개임;
- `K`가 여러 구현 요소를 함께 바꾸므로 구성요소별 인과효과는 식별되지 않음;
- `T`나 `K`를 offline하게 선택하는 규칙은 아직 없음;
- ideal proximal statement와 live one-step neural optimizer 사이의 직접적 보증은 없음.

따라서 원고의 강한 결론은 **finite-step routing과 refinement depth가 명시된 동일가중 high-budget aggregate에서 tested stability envelope를 재현 가능하게 확장한다**는 것입니다. Ideal Wasserstein 분석은 그 결과를 대체하는 보증이 아니라, 관측된 구조에 대한 국소 기하 설명입니다.
