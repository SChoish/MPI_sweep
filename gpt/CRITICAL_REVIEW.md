# MPI AISTATS 2027 원고 비평적 읽기

## 1. 총평

현재 원고의 가장 강한 기여는 **완전한 fixed-budget phase diagram에서 multi-hop actor bundle이 large-budget failure의 하위 꼬리를 줄인다는 실증 결과**입니다. 이 결과는 단순 평균 개선보다 훨씬 흥미롭습니다. 특히 새로 완료된 proximal `K=4` grid는 `T >= 4` 평균을 `25.97 (K=1) -> 63.61 (K=4)`로 높이고, score `<20`인 two-seed cell을 `35/63 -> 9/63`으로 줄입니다. `K=3 -> K=4`의 평균 증분은 `+10.83`이지만 중앙값은 `+0.32`이므로, 효과의 본질은 전반적인 평행 이동이 아니라 **collapse rescue**입니다.

반면 현재 원고는 이 강한 실증 결과보다 **target-policy displacement / deployment reach의 분리와 Wasserstein proximal 해석을 더 강한 메커니즘 주장처럼 전면에 배치**합니다. 현재 artifact와 component control이 실제로 지지하는 범위는 그보다 좁습니다. 원고를 그대로 제출하면 리뷰어가 다음과 같이 읽을 가능성이 큽니다.

> 결과는 흥미롭지만, 무엇이 원인인지 분리되지 않았고, 구현은 정확한 proximal/JKO solve가 아니며, `K`에 따라 critic initialization까지 달라진다. 이론은 조건부이거나 항등식에 가깝고, empirical claim과 직접 연결되지 않는다.

따라서 논문의 중심을 **“기하학적 implicit method”**에서 **“고정 actor budget을 여러 persistent branch로 나누면 coupled offline actor–critic training의 안정성 영역이 넓어진다”**로 옮기는 것이 낫습니다. GPT 원고는 이 방향으로 재구성했습니다.

---

## 2. 강점

### 2.1 phase diagram이 단일 최종점 benchmark보다 정보량이 많음

9 environments × 14 budgets × 2 seeds의 complete grid는 단순히 최적 hyperparameter를 골라 평균을 비교하는 실험보다 훨씬 설득력이 있습니다. 어떤 환경에서는 완만한 response가, 어떤 환경에서는 급격한 bifurcation-like collapse가 나타난다는 점을 보여줍니다. 특히 평균과 collapse count를 함께 제시하여 “조금씩 전부 좋아졌다”와 “실패를 구조적으로 줄였다”를 구분한 점이 좋습니다.

### 2.2 이론의 적용 범위를 상당 부분 정직하게 제한함

원고는 learned critic에 대한 descent와 true return improvement를 구분하고, fixed-critic local expansion과 moving-critic large-budget regime을 분리하려고 합니다. 이는 초기 MPI 원고보다 훨씬 정교해진 부분입니다.

### 2.3 explicit/linearized control이 proximal loss 특수성 주장을 약화시키면서도 핵심 결과를 보강함

`Lin K=2 -> K=3`에서도 high-budget mean이 `36.68 -> 43.91`, collapse가 `29/63 -> 24/63`으로 개선됩니다. 이 결과는 “implicit proximal objective 자체가 유일한 원인”이라는 주장을 막지만, 오히려 더 일반적인 **substepped actor-chain bundle**의 안정성 효과를 지지합니다.

### 2.4 새 K=4 결과는 논문의 empirical value를 분명히 강화함

새 complete `K=4` grid는 high-budget mean `63.61`, tail mean `56.26`, collapse `9/63`입니다. `K=3`에서 collapsed였던 8개 cell을 stable로 옮기고, 반대 방향의 cell-level transition은 0개입니다. 이 결과는 기존 원고의 “K=3가 가장 강하다”는 서술을 낡게 만들지만, 핵심 현상 자체는 더 강하게 만듭니다.

### 2.5 K=1 추가 seed가 baseline failure landscape를 재현함

새 seeds 2–3에서 high-budget mean은 `24.79`, collapsed cells는 `35/63`으로, 기존 seeds 0–1의 `25.97`, `35/63`과 거의 같습니다. 따라서 one-hop collapse landscape 자체는 특정 두 seed의 우연으로 보기 어렵습니다.

---

## 3. 중대한 문제

## 3.1 [가장 중요] `K`에 따라 critic initialization이 달라지는 confound

현재 `train_td3bc.py`의 초기화는 다음 구조입니다.

```python
keys = jax.random.split(rng, mpi_steps + 1)
actors = initialize(keys[:-1])
critic = initialize(keys[-1])
```

`jax.random.split(key, n)`의 마지막 key는 `n`에 따라 달라집니다. 따라서 같은 integer seed라도 `K=1,2,3,4`는 **동일한 critic initialization을 공유하지 않습니다**. actor–critic collapse를 critic reliability로 설명하는 논문에서 이는 사소한 구현 차이가 아니라 핵심 internal-validity threat입니다.

현재의 “paired seeds”라는 표현은 evaluation seed와 seed label 수준에서는 맞지만, critic initialization에 대해서는 맞지 않습니다. 새로운 원고에서는 이를 명시적으로 limitations에 넣었습니다. 최종 제출 전에는 다음 형태의 rerun이 가장 중요합니다.

```python
actor1_key, critic_key, aux_key = jax.random.split(seed_key, 3)
aux_actor_keys = jax.random.split(aux_key, K - 1)
```

이렇게 하면 actor 1과 critic은 모든 K에서 동일하게 유지됩니다.

## 3.2 구현은 exact proximal/JKO step이 아님

수학적 정의는

```text
mu_k in argmin proximal objective
```

이지만 실제 코드는 각 delayed actor update에서 actor마다 **Adam 한 번**만 적용합니다. 더구나 later actors는 predecessor의 parameter copy로 초기화되지 않고 독립적으로 초기화된 persistent network입니다. 따라서 실제 동역학은 다음이 혼합된 amortized multi-network training입니다.

- proximal-style loss
- persistent optimizer state
- independent actor initialization
- one gradient update per hop
- moving critic
- minibatch noise

그러므로 “implicit integrator”, “JKO update”, “backward Euler realization”이라는 명칭은 과합니다. 가장 안전한 명칭은 **proximal-loss realization (MPI-Prox)**입니다. 이론은 이상화된 operator를 설명하고, 구현은 그 objective를 한 번 업데이트하는 근사라고 분리해야 합니다.

## 3.3 Proposition 2는 구현 보장이 아니라 조건부 comparator inequality

현재 finite-step margin은

```text
computed objective <= comparator objective + epsilon
```

을 가정한 뒤 부등식을 재배열한 것입니다. 수학적으로 맞지만, 실제 one-Adam-step이 이 조건을 만족하는지는 로그에서 검증되지 않습니다. 따라서 “finite-step guarantee”보다 **conditional learned-critic descent inequality**라고 부르는 것이 정확합니다.

첫 hop에서는 sampled dataset action이 deterministic policy class의 feasible comparator가 아니라는 점도 중요합니다. 원고가 이를 언급하지만, 이 사실은 본문 한가운데의 단서가 아니라 proposition의 적용 범위에 직접 포함되어야 합니다.

## 3.4 critic-error transfer는 중요한 경고지만 이론적 novelty는 제한적

`Qhat = Qdagger + e`를 대입해 error difference를 분리하는 식은 유용한 해석이지만 본질적으로 algebraic identity입니다. 이를 큰 theorem처럼 전면에 두기보다, “learned-critic descent가 true objective로 넘어가려면 pathwise error control이 필요하다”는 명확한 caveat로 사용하는 편이 낫습니다.

## 3.5 target-routing component evidence가 headline claim을 충분히 지지하지 않음

shadow-critic experiment는 final-route critic을 **first-route continuation의 simulator return**과 비교합니다. 이 estimand는 구조적으로 first route와 정렬되어 있습니다. 원고가 이 한계를 적어두었지만, abstract와 introduction에서 “final route가 더 나쁘다”는 식으로 전면에 내세우면 리뷰어가 estimand mismatch를 바로 지적할 수 있습니다.

또한 fixed-reference K=3가 sequential K=3와 거의 같은 결과를 보였으므로, 현재 component control은 re-centering 자체가 원인이라는 주장을 지지하지 않습니다. 느린 target Polyak만으로 rescue가 재현되지 않았다는 결과도 유용하지만, routing의 causal return effect를 보여주지는 않습니다.

결론은 다음 수준이어야 합니다.

> first-hop routing은 architecture를 정의하며, targeted diagnostics와 양립한다. 그러나 sweep-wide gain의 원인을 routing이나 re-centering 하나로 분해하지는 못했다.

## 3.6 complete grid가 compute matched가 아님

K-hop condition은 delayed actor update마다 K번의 actor optimizer call을 수행합니다. 따라서 K가 커질수록 actor compute와 parameter count가 증가합니다. 작은 targeted control은 있으나 complete-grid compute-matched baseline은 없습니다. 원고의 main estimand를 “implemented bundle”이라고 부르는 것은 맞지만, 제목과 abstract에서도 같은 수준으로 제한해야 합니다.

## 3.7 raw logs / checkpoints / diagnostic scripts가 repository에 없음

root `.gitignore`는 `logs/`, `results/`, `*.pkl` 등을 제외합니다. 현재 repository에는 final score matrices와 hard-coded figure aggregates는 있지만, 다음 결과를 독립적으로 재계산할 raw artifact는 없습니다.

- target-exposure 88/90
- common-critic gain
- simulator calibration error
- route-only shadow critic
- fixed-reference / direct-final controls
- frozen-critic local audit

따라서 기존 reproducibility checklist의 “diagnostic scripts와 controls를 포함한다”는 문구는 현재 repository snapshot 기준으로 과장입니다. GPT bundle은 이 결과를 secondary inherited evidence로만 취급하고, 직접 재계산 가능한 sweep을 headline으로 삼았습니다.

## 3.8 two-seed cross-K grid와 correlated cell counting

63개의 environment–budget cell은 독립 표본이 아닙니다. 동일한 환경, dataset, seed label, architecture를 공유합니다. 따라서 8개 rescue / 0개 harm이나 collapse count를 일반적인 binomial evidence처럼 해석하면 안 됩니다. descriptive phase diagram으로 제시하는 것은 괜찮지만, p-value나 독립표본 confidence interval로 확장해서는 안 됩니다.

## 3.9 기존 five-seed extension table은 본문에서 빼는 것이 나음

IQL/ReBRAC/TD3+BC의 original extension table은 보기에는 강하지만, launch grid, model-selection rule, configuration, raw artifact가 현재 bundle에 없습니다. 새로운 fixed-budget story와도 직접 연결되지 않습니다. appendix의 historical context로 남기거나 완전히 빼는 것이 논문의 신뢰도를 높입니다.

---

## 4. 원고 구성과 문장에 대한 비평

### 4.1 제목

현재 제목은 지나치게 길고 target/deployment mechanism을 확정된 핵심처럼 보이게 합니다.

기존:

> Multi-Step Proximal Policy Improvement: Separating Target-Policy Displacement from Deployment Reach in Offline Actor–Critic Learning

권장:

> Multi-Step Proximal Policy Improvement: Budget Splitting for Stable Offline Actor–Critic Learning

새 제목은 실제로 가장 강하게 지지되는 empirical claim을 전면에 둡니다.

### 4.2 Abstract

기존 abstract는 theorem, exposure, diagnostics, shadow critics를 한꺼번에 넣어 밀도가 너무 높습니다. 또한 가장 재현 가능한 완전 sweep보다 mechanism claim이 앞섭니다. 새 abstract는 다음 순서로 바꿨습니다.

1. 문제와 architecture
2. 이론의 제한된 범위
3. K=4 complete-grid headline
4. linearized control과 K=1 replication
5. causal limitation과 initialization confound

### 4.3 Introduction

현재 introduction은 이미 많은 caveat를 포함해 정직하지만, 네 번째 문단부터 local geometry, finite step, large-budget coupling이 한꺼번에 들어와 독자가 핵심 empirical question을 잃기 쉽습니다. 새로운 버전은 먼저 phase-diagram result를 설명하고, 그 다음에 “왜 이것이 mechanism proof가 아닌지”를 명시합니다.

### 4.4 Method

현재 method는 수학적 operator와 실제 code path를 충분히 분리하지 않습니다. 특히 “maintains K independently initialized persistent actors”와 “mu_k is proximal argmin”이 가까이 놓여 있어 독자가 구현도 argmin을 푼다고 오해할 수 있습니다. 새 버전은 `Implementation is not an exact proximal solve`라는 remark를 본문에 명시했습니다.

### 4.5 Experiments

현재 원고의 실험부는 좋은 결과가 많지만, main story가 여러 진단으로 분산됩니다. 새로운 버전은 실험을 다음 우선순위로 정리했습니다.

1. complete 6-method sweep (`K=4` 포함)
2. K=3 -> K=4 lower-tail rescue
3. Lin K=2 -> K=3
4. K=1 independent seed-pair replication
5. inherited diagnostics의 evidential status
6. code-audit limitations

### 4.6 Layout

업로드된 PDF는 읽을 수 있으나 다음 문제가 있습니다.

- 첫 페이지 제목이 길고 abstract가 과밀함
- 상단 heavy rule과 running header가 너무 가까움
- main text가 8페이지 안에 과도하게 압축되고, 뒤 appendix에는 빈 공간이 큼
- 일부 표와 figure가 너무 작아 수치 확인이 어려움

새 bundle은 제목을 줄이고, 상단 여백과 title block을 조정하며, main table/figures의 수를 줄였습니다.

---

## 5. 새 GPT 원고에서 반영한 결정

- `K=4` complete grid를 main table과 main figure에 포함
- `K=3가 strongest`라는 낡은 문구 삭제
- proximal family의 high-budget mean과 collapse를 K=1..4로 제시
- `K=3 -> K=4`: mean `+10.83`, median `+0.32`, rescue `8`, harm `0`을 lower-tail evidence로 제시
- `Lin K=3` complete result 유지
- K=1 seeds 2–3 replication을 appendix에 추가
- exact JKO/implicit integrator 표현을 완화하고 `proximal-loss realization` 사용
- finite-step theorem을 conditional inequality로 재명명
- route-only 및 simulator diagnostics를 abstract headline에서 제거
- raw artifact가 없는 진단은 secondary inherited evidence로 명시
- `K`-dependent critic initialization을 main limitation으로 명시
- original IQL/ReBRAC extension table 제거
- standalone build script, figure script, data snapshot, claim ledger 포함

---

## 6. 제출 전 우선순위

1. **K-invariant critic/actor-1 initialization으로 K=1..4 핵심 grid rerun**
2. K=2/3 추가 seeds 완료 후 최소 high-budget subset에서 4-seed comparison
3. complete-grid compute-matched control 또는 적어도 representative environment의 pre-registered subset
4. later actor를 independent init과 predecessor copy init으로 비교
5. conditional descent premise를 실제 training log에서 audit
6. raw diagnostics, configs, checkpoints, analysis scripts를 artifact에 포함
7. 공식 AISTATS 2027 style 공개 시 provisional style 교체

---

## 7. 예상 평가

현재 원고 그대로라면, empirical novelty는 충분하지만 mechanism/theory framing과 artifact gap 때문에 **borderline**으로 평가될 가능성이 있습니다. 반대로 K=4 result를 포함하고, claim을 end-to-end stability bundle로 낮추며, initialization confound를 고친 rerun을 추가하면 논문의 설득력은 크게 올라갑니다.

제 판단을 수치화하면 다음과 같습니다.

| 항목 | 현재 원고 | GPT 재구성 방향 |
|---|---:|---:|
| 문제 설정 / 중요성 | 7.5/10 | 8.0/10 |
| empirical evidence | 7.5/10 | 8.5/10 |
| 이론적 novelty | 5.5/10 | 5.5/10 |
| claim–evidence 정합성 | 5.5/10 | 8.0/10 |
| 재현성 | 4.5/10 | 6.5/10 |
| 명료성 | 6.0/10 | 8.0/10 |

GPT 원고의 목적은 더 강한 말을 쓰는 것이 아니라, **현재 artifact가 실제로 가장 강하게 지지하는 논문을 만드는 것**입니다.
