# BAR 원고 통합 비평 - AAAI 2026 GPT revision

## 정정

이전 GPT 비평에서 `K`에 따라 critic initialization subkey가 달라진다는 사실을 **중대한 confound**로 규정하고, shared-initialization rerun을 필수로 제안한 것은 잘못이었습니다. 각 방법의 목표가 randomized training procedure의 marginal mean

\[
\mu_K=\mathbb E_{\omega\sim P_K}[Y_K(\omega)]
\]

이라면 각 K에서 독립적인 유효 난수 표본으로 평균을 추정하면 충분합니다. 동일 initialization은 common-random-number blocking으로 차이 추정의 분산을 줄일 수 있지만 validity의 필요조건이 아닙니다. 이 정정은 `K` 자체가 의미 없다는 뜻이 아닙니다. K는 여전히 complete phase diagram의 refinement-depth 변수이며, K=4 결과도 중심 증거로 유지했습니다.

## 총평

현재 연구의 가장 강한 기여는 9개 D4RL 데이터셋, 14개 nominal coefficient budget, 2개 seed에 걸친 complete stability phase diagram입니다. K=1에서 K=4로 갈수록 high-budget mean이 25.97에서 63.61로 증가하고 score<20 cell이 35/63에서 9/63으로 감소합니다. K4-K1의 task-cluster interval은 0을 명확히 벗어나므로 broad rescue는 설득력이 있습니다. 반면 K4-K3의 collapse-risk interval은 0에 닿으므로, “K4가 K3보다 collapse probability를 확실히 낮춘다”는 표현은 피해야 합니다.

원고의 가장 중요한 한계는 데이터 오류가 아니라 **ideal proximal analysis와 실제 one-Adam-step persistent actor chain 사이의 간극**입니다. 따라서 정확한 정체성은 다음과 같습니다.

> BAR is an empirical fixed-nominal-budget refinement bundle whose behavior is locally motivated, but not certified, by the proximal analysis.

## 외부 감사 보고서 반영 여부

이번 수정에서는 보고서의 네 P0 항목을 모두 실제 소스에 반영했습니다.

### 1. C_k 구현 불일치

기존의 단일 정의를 폐기하고 경로별 정의를 만들었습니다.

- Prox hop 1: pre-update actor output에서 Q scale 계산
- Prox hop k>=2: predecessor/reference actor output에서 계산
- Lin hop 1: dataset action에서 계산
- Lin hop k>=2: predecessor/reference actor output에서 계산

따라서 BAR-Prox와 BAR-Lin은 action metric coefficient는 맞추지만 hop-1 Q-normalization point는 같지 않다고 명시했습니다.

### 2. 평가 주기

5,000을 50,000 updates로 수정했고, 모든 headline cell은 1M checkpoint의 10-episode mean임을 명시했습니다.

### 3. Taylor remainder

C2만으로 O(beta^3)를 쓰지 않습니다. Hessian locally Lipschitz(C3 suffices)를 가정해 O(beta^3)를 유지하고, C2만 가정하면 weaker remainder가 필요하다고 적었습니다.

### 4. dataset 이름

보고서의 `medium-expert-v2` 제안은 구현과 충돌합니다. 실제 data loader는 `halfcheetah-expert-v2`, `hopper-expert-v2`, `walker2d-expert-v2`를 사용하므로 이를 정확한 ID로 통일했습니다. 표의 E는 이 expert-v2를 뜻한다고 명시했습니다.

## 이론 평가

대수적 명제 자체는 맞습니다. 다만 conditional comparator premise가 live Adam update에서 성립하는지는 측정하지 않았습니다. First hop에서는 sampled dataset action이 deterministic policy comparator가 아니라는 문제도 있습니다. Fixed-K composition은 `Psi_h(x)=x+h f(x)+...`와 `Psi_0=id`를 가정하지만, persistent actor가 predecessor와 이미 다르면 h=0에서도 anchoring gradient가 남을 수 있습니다. 이론은 삭제할 필요는 없지만 operator-level motivation으로 내려야 합니다.

## 실증 평가

### Complete sweep

- K4-K1 high-budget mean difference: +37.64, dataset-bootstrap 95% [20.46,54.27], 8/9 datasets positive.
- K4-K3 mean difference: +10.83, [2.26,22.04], 6/9 positive.
- K4-K1 raw collapse-risk difference: -0.310, [-0.484,-0.143].
- K4-K3 raw collapse-risk difference: -0.095, [-0.206,0].

이는 K4의 broad improvement를 지지하지만 incremental collapse claim은 약하게 만듭니다. K4-K3 cellwise median이 .32라는 사실은 평균 gain이 lower-tail rescue에 집중되어 있음을 보여줍니다.

### Target movement

P3 target-action displacement는 TD3보다 88/90 cell에서 작고 median ratio는 .658입니다. 이 결과는 cluster bootstrap에서도 강합니다. 그러나 final proxy는 current state/online actor/dataset current action을 사용하고 target metric은 next state/Polyak actor/paired next action을 사용합니다. 두 수치를 동일 거리의 두 branch처럼 직접 비교해서는 안 됩니다.

### Failure/simulator

Failure signature는 재현되지만 일부 TD residual quantile은 raw per-state array가 아니라 archived run summary에서 옵니다. Simulator의 기존 -38.7은 pooled-state median입니다. revised paper는 checkpoint를 단위로 한 -41.4를 사용합니다.

### Route shadow

Primary four-checkpoint average에서는 8/8이지만 final checkpoint only에서는 5/8입니다. 평균 .124는 한 개의 .803 delta에 강하게 영향받고 median은 .013입니다. 게다가 cell은 post hoc입니다. 따라서 `p=.0078`을 confirmatory causal evidence처럼 쓰지 않고 targeted descriptive alignment result로 제한했습니다.

## 재현성

새 supplement는 다음을 한 페이지 수준으로 명시합니다.

- actor/critic architecture와 activation
- optimizer와 framework-default initialization
- target initialization
- state/reward preprocessing
- target noise와 clipping
- exact dataset IDs
- terminal/timeout handling
- 50k evaluation 및 1M aggregation
- current/next-state displacement 정의
- rho_1 및 symmetric relative error
- TD p99 정의
- common T=1 critic provenance
- bootstrap unit/iterations
- time and memory scaling O(K)

Checklist는 과도한 Yes를 낮추거나 supplement 근거를 보강했습니다.

## 최종 판단

수정 전 원고는 읽히지만 수학/실증 scope가 완전히 닫히지 않은 7.5-8/10 수준이라는 외부 평가가 타당했습니다. 이번 GPT revision은 명백한 P0 오류와 aggregation ambiguity를 제거했고, task-cluster uncertainty와 reproducibility disclosure를 추가했습니다. 여전히 complete-grid compute matching이나 live objective-descent audit이 없으므로 causal/mechanistic paper로 과장해서는 안 됩니다. 그러나 **강한 descriptive bundle-effect paper**로서는 제출 가능한 구조에 가까워졌습니다.
