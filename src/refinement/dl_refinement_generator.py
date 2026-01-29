"""src.refinement.dl_refinement_generator

논문 SDT에서의 핵심 구성요소 중 하나인 "Concept Refinement" 후보(Refinements)를 생성한다.

이번 버전은 다음을 목표로 한다:
1) 하드코딩 임계값 제거: 노드(현재 인스턴스 집합) 기반으로 numeric cut-point를 동적으로 생성
2) (정합성/성능) 정적 refinement(Existential/Boolean) 캐싱
3) FunctionalGroup ABox를 "개체(Individual)"로 두는 방식과 호환

주의: OWL reasoner 기반 질의로 만족 여부를 판정하는 완전한 DL-Query 기반 구현은
추가 작업이 필요하다. 현재는 파이썬에서 ABox 값을 직접 조회하여 판정한다.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from owlready2 import *

logger = logging.getLogger(__name__)


class DLRefinement:
    """Refinement predicate (used as a split test in SDT)."""

    def __init__(
        self,
        refinement_type: str,
        property_name: Optional[str] = None,
        target_class: Optional[str] = None,
        value=None,
        operator: Optional[str] = None,
    ):
        # refinement_type in {'existential', 'value', 'numeric'}
        self.type = refinement_type
        self.property = property_name
        self.target = target_class
        self.value = value
        self.operator = operator

    def signature(self) -> Tuple:
        return (self.type, self.property, self.target, self.operator, self.value)

    def __repr__(self) -> str:
        if self.type == "existential":
            return f"∃{self.property}.{self.target}"
        if self.type == "value":
            return f"{self.property}={self.value}"
        if self.type == "numeric":
            return f"{self.property}{self.operator}{self.value}"
        return "Unknown"

    def __hash__(self):
        return hash(self.signature())

    def __eq__(self, other):
        if not isinstance(other, DLRefinement):
            return False
        return self.signature() == other.signature()

    # --- Optional OWL conversion helpers (not used by current learner) ---
    def to_owlready_restriction(self, onto):
        """Best-effort conversion to an Owlready2 class expression.

        - existential: ObjectProperty some Class
        - value: DataProperty value literal
        - numeric: DataProperty some DatatypeRestriction (OWL2) (best-effort)
        """

        prop = getattr(onto, self.property, None)
        if prop is None:
            return None

        if self.type == "existential":
            target_cls = getattr(onto, self.target, None)
            if target_cls is None:
                return None
            return prop.some(target_cls)

        if self.type == "value":
            return prop.value(self.value)

        if self.type == "numeric":
            # Numeric comparisons belong to OWL2 datatype restrictions.
            # Owlready2 supports ConstrainedDatatype in recent versions.
            try:
                base_dt = float if isinstance(self.value, float) else int
                if self.operator == "<=" and self.value is not None:
                    dt = ConstrainedDatatype(base_dt, max_inclusive=self.value)
                elif self.operator == ">=" and self.value is not None:
                    dt = ConstrainedDatatype(base_dt, min_inclusive=self.value)
                elif self.operator == "==" and self.value is not None:
                    return prop.value(self.value)
                else:
                    return None
                return prop.some(dt)
            except Exception:
                return None

        return None


class RefinementGenerator:
    """Generate refinement candidates, optionally using caching."""

    def __init__(self, onto):
        self.onto = onto
        self._cached_existential: Optional[List[DLRefinement]] = None
        self._cached_boolean: Optional[List[DLRefinement]] = None
        self._cache: Dict[Tuple, List[DLRefinement]] = {}

    def generate_all_refinements(
        self,
        *,
        center_signature: str = "Molecule",
        instances: Optional[Sequence] = None,
        exclude: Optional[Set[DLRefinement]] = None,
        enable_cache: bool = True,
        max_numeric_thresholds_per_prop: int = 15,
    ) -> List[DLRefinement]:
        """Generate refinements for a given node.

        - Existential + Boolean are cached (schema-derived).
        - Numeric refinements are generated from *instances* (node-specific).
        """

        exclude = exclude or set()
        instances = list(instances or [])

        cache_key = None
        if enable_cache:
            # Note: node-specific numeric thresholds depend on the instance set.
            # We still cache by (center_signature, n_instances, excluded_signatures)
            # which helps when identical subsets appear (or in repeated calls).
            cache_key = (
                center_signature,
                len(instances),
                tuple(sorted([e.signature() for e in exclude])),
                max_numeric_thresholds_per_prop,
            )
            if cache_key in self._cache:
                return self._cache[cache_key]

        refinements: List[DLRefinement] = []

        existential = self._get_existential_refinements()
        boolean = self._get_boolean_refinements()
        numeric = self._generate_numeric_refinements(
            instances, max_thresholds_per_prop=max_numeric_thresholds_per_prop
        )

        for r in existential + boolean + numeric:
            if r not in exclude:
                refinements.append(r)

        if enable_cache and cache_key is not None:
            self._cache[cache_key] = refinements

        return refinements

    def _get_existential_refinements(self) -> List[DLRefinement]:
        if self._cached_existential is not None:
            return self._cached_existential

        refinements: List[DLRefinement] = []

        for obj_prop in self.onto.object_properties():
            if obj_prop.name != "containsFunctionalGroup":
                continue

            fg_class = getattr(self.onto, "FunctionalGroup", None)
            if fg_class is None:
                continue

            for subclass in fg_class.descendants():
                if subclass == fg_class:
                    continue
                refinements.append(
                    DLRefinement(
                        refinement_type="existential",
                        property_name=obj_prop.name,
                        target_class=subclass.name,
                    )
                )

        self._cached_existential = refinements
        return refinements

    def _get_boolean_refinements(self) -> List[DLRefinement]:
        if self._cached_boolean is not None:
            return self._cached_boolean

        refinements: List[DLRefinement] = []

        # Auto-detect boolean data properties
        for dp in self.onto.data_properties():
            if dp.name in {"hasLabel", "hasSMILES"}:
                continue

            rng = list(getattr(dp, "range", []) or [])
            if bool not in rng:
                continue

            refinements.append(
                DLRefinement(
                    refinement_type="value",
                    property_name=dp.name,
                    value=True,
                    operator="==",
                )
            )
            refinements.append(
                DLRefinement(
                    refinement_type="value",
                    property_name=dp.name,
                    value=False,
                    operator="==",
                )
            )

        self._cached_boolean = refinements
        return refinements

    def _generate_numeric_refinements(
        self,
        instances: Sequence,
        *,
        max_thresholds_per_prop: int = 15,
    ) -> List[DLRefinement]:
        """Node-specific numeric cutpoints (no hardcoded thresholds)."""

        refinements: List[DLRefinement] = []

        if not instances:
            return refinements

        for dp in self.onto.data_properties():
            if dp.name in {"hasLabel", "hasSMILES"}:
                continue

            rng = list(getattr(dp, "range", []) or [])
            is_numeric = (int in rng) or (float in rng)
            if not is_numeric:
                continue

            values: List[float] = []
            for inst in instances:
                v = getattr(inst, dp.name, None)
                if v is None:
                    continue
                # Owlready2 FunctionalProperty returns scalar
                try:
                    values.append(float(v))
                except Exception:
                    continue

            if len(values) < 2:
                continue

            unique_vals = np.unique(np.array(values, dtype=float))
            if unique_vals.size < 2:
                continue

            thresholds: List[float] = []
            if unique_vals.size <= 30:
                # Midpoints between unique sorted values
                unique_sorted = np.sort(unique_vals)
                mids = (unique_sorted[:-1] + unique_sorted[1:]) / 2.0
                thresholds = mids.tolist()
            else:
                qs = np.linspace(0.1, 0.9, 9)
                thresholds = np.quantile(np.array(values, dtype=float), qs).tolist()

            # Deduplicate and cap
            thresholds = sorted(set([float(t) for t in thresholds]))
            if len(thresholds) > max_thresholds_per_prop:
                # Evenly sample capped thresholds
                idxs = np.linspace(0, len(thresholds) - 1, max_thresholds_per_prop)
                thresholds = [thresholds[int(i)] for i in idxs]

            for thr in thresholds:
                refinements.append(
                    DLRefinement(
                        refinement_type="numeric",
                        property_name=dp.name,
                        value=float(thr),
                        operator="<=",
                    )
                )

        return refinements

    def filter_valid_refinements(
        self, refinements: List[DLRefinement], instances: Sequence
    ) -> List[DLRefinement]:
        """Keep only refinements that split the current instances (0 < sat < n)."""

        instances = list(instances)
        n = len(instances)
        if n == 0:
            return []

        valid: List[DLRefinement] = []
        for ref in refinements:
            satisfying = self.count_satisfying_instances(ref, instances)
            if 0 < satisfying < n:
                valid.append(ref)
        return valid

    def count_satisfying_instances(
        self, refinement: DLRefinement, instances: Sequence
    ) -> int:
        count = 0
        for inst in instances:
            if self.instance_satisfies_refinement(inst, refinement):
                count += 1
        return count

    def instance_satisfies_refinement(self, instance, refinement: DLRefinement) -> bool:
        """Check refinement satisfaction against ABox values.

        This assumes FunctionalGroup grounding uses individuals:
            mol.containsFunctionalGroup = [Amine('FG_...'), Nitro('FG_...'), ...]
        """

        if refinement.type == "existential":
            prop_values = getattr(instance, refinement.property, [])
            target_class = getattr(self.onto, refinement.target, None)
            if target_class is None:
                return False
            for val in prop_values:
                if isinstance(val, target_class):
                    return True
            return False

        if refinement.type == "value":
            prop_value = getattr(instance, refinement.property, None)
            return prop_value == refinement.value

        if refinement.type == "numeric":
            prop_value = getattr(instance, refinement.property, None)
            if prop_value is None:
                return False
            try:
                v = float(prop_value)
                thr = float(refinement.value)
            except Exception:
                return False

            if refinement.operator == "<=":
                return v <= thr
            if refinement.operator == ">=":
                return v >= thr
            if refinement.operator == "==":
                return v == thr
            return False

        return False


if __name__ == "__main__":
    from src.ontology.ontology_loader import OntologyLoader

    loader = OntologyLoader("ontology/bbbp_ontology.owl")
    onto = loader.load()

    # Note: numeric refinements require instances
    molecules = list(getattr(onto, "Molecule").instances())
    generator = RefinementGenerator(onto)
    refinements = generator.generate_all_refinements(
        center_signature="Molecule",
        instances=molecules[:200],
        enable_cache=True,
    )

    print(f"\nGenerated {len(refinements)} refinements")
    print("\nSample refinements:")
    for ref in refinements[:15]:
        print(f"  {ref}")
