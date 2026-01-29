# Changelog

> 이 파일은 주요 변경사항(특히 논문 SDT 정합성 강화)을 사람이 읽기 쉬운 형태로 기록합니다.

## 2026-01-29 — SDT-v0.1.2 (paper-hardening)

이전 push(`SDT-v0.1.1_2026-01-29-AUC89`) 대비 **논문 SDT 구현 정합성/평가 공정성/성능(속도)** 측면을 강화했습니다.

### 핵심 차별점

- **(1) Functional Group ABox grounding 방식 변경 (Class-link → Individual-link)**
  - 기존: `mol.containsFunctionalGroup.append(fg_class)` 처럼 *클래스 자체*를 연결하는 형태
  - 변경: `FG_{idx}_{fg}` 개체를 생성하고 `mol.containsFunctionalGroup.append(fg_ind)`로 링크
  - 효과: `∃ containsFunctionalGroup.Amine` 같은 DL 스타일 refinement를 **ABox 기반으로 정확히 평가** 가능

- **(2) Reasoner 기반 semantic enrichment을 위한 Defined Class 추가** (`src/ontology/bbbp_owl_generator.py`)
  - 예: `AromaticMolecule ≡ Molecule ⊓ hasAromaticRing value True`
  - 예: `LipinskiCompliant ≡ Molecule ⊓ obeysLipinskiRule value True`
  - 예: `HasAmine ≡ Molecule ⊓ containsFunctionalGroup some Amine`
  - 효과: reasoner 실행 시 inferred typing을 통해 **의미 기반 분할 후보**를 더 자연스럽게 지원

- **(3) 하드코딩 threshold 제거: 노드 인스턴스 기반 dynamic numeric cutpoint 생성** (`src/refinement/dl_refinement_generator.py`)
  - 기존: `hasRingCount >= 2` 같은 임계값 리스트를 코드에 고정
  - 변경: 현재 노드의 `instances` 분포로부터 mid-point/quantile 기반 cutpoint를 생성 (상한 cap 적용)
  - 효과: 데이터/노드에 맞는 수치 분할을 생성 → 논문 스타일(노드별 refinement 생성)에 더 근접

- **(4) Refinement 캐싱 도입 (정합성 유지 + 속도 개선)** (`src/refinement/dl_refinement_generator.py`)
  - schema-derived refinement(Existential/Boolean) 캐시
  - 노드 단위 전체 후보도 `(center_signature, n_instances, exclude, ...)` 키로 캐시

- **(5) 평가 leakage 방지: train-only fit 강제/권장** (`src/sdt/sdt_learner.py`, `experiments/true_sdt_experiment.py`)
  - `learner.fit(..., instances=train_instances)` 형태로 학습 데이터를 명시
  - `instances` 미지정 시 경고 출력(후방 호환은 유지)

- **(6) 동적 center concept 추적(부분 구현)** (`src/sdt/sdt_learner.py`)
  - `SDTNode.path_conditions`로 root→node 경로의 refinement(+/-)를 저장
  - `SDTNode.center_signature`로 노드별 "center concept"을 문자열 서명으로 구성
  - `center_expr`는 Owlready2 restriction으로 가능한 범위에서 best-effort로 누적(데이터타입 제약은 제한)

- **(7) 실행 과정 가시성 개선: print 기반 progress logging** (`experiments/true_sdt_experiment.py`, `src/sdt/sdt_learner.py`)
  - reasoner/학습/평가 진행상황을 timestamped print로 확인 가능

### 파일별 변경 요약

- `src/ontology/bbbp_owl_generator.py`
  - FG individual grounding
  - Defined classes 추가
  - 불필요 import 정리

- `src/refinement/dl_refinement_generator.py`
  - numeric cutpoint 동적 생성
  - 정적/전체 후보 캐싱
  - FG individual ABox 만족도 판정과 호환

- `src/sdt/sdt_learner.py`
  - train-only 학습 지원(`fit(..., instances=...)`)
  - node-aware candidate generation(exclude 경로에서 사용한 refinement)
  - progress callback 지원

- `experiments/true_sdt_experiment.py`
  - reasoner 실행 포함
  - train/test split 후 train-only 학습
  - 평가 진행 출력

- `ontology/bbbp_ontology.owl`
  - 위 스키마/grounding 반영하여 재생성됨 (추적 파일이므로 변경 포함)

### 재현 방법

- 실험 실행: `python experiments/true_sdt_experiment.py`

> 참고: 현재 구현은 ABox 직접 조회로 만족 여부를 판정하는 "best-effort" DL 구현입니다.
> 완전한 논문 수준의 DL-query/Reasoner 기반 만족도 판정 및 center concept의 reasoner-driven 동적 갱신은 후속 작업(1~7 중 남은 부분)에 해당합니다.
