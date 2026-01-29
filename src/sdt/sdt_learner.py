"""
논문 SDT 학습 알고리즘
Reasoner 기반 Center Class 동적 갱신 + DL Refinement 적용
"""

from owlready2 import *
from typing import Callable, List, Dict, Tuple, Optional
import numpy as np
import types
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.refinement.dl_refinement_generator import (
    RefinementGenerator,
    DLRefinement,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SDTNode:
    """논문 SDT의 노드 (Center Class 기반)"""
    
    def __init__(
        self,
        center_class,
        center_class_name: str,
        instances: List,
        center_expr=None,
        depth: int = 0,
        node_id: int = 0,
        path_conditions: Optional[List[Tuple[DLRefinement, bool]]] = None,
    ):
        self.center_class = center_class
        self.center_class_name = center_class_name
        # Best-effort OWL expression of the path concept.
        self.center_expr = center_expr if center_expr is not None else center_class
        self.instances = instances
        self.depth = depth
        self.node_id = node_id

        # Conjunction of refinements along the root->node path.
        # True: satisfied branch, False: NOT satisfied branch.
        self.path_conditions = list(path_conditions or [])

        # Split test and structure
        self.refinement = None
        self.is_leaf = False
        self.predicted_label = None

        self.left_child = None  # refinement 만족
        self.right_child = None  # refinement 불만족

        # Stats
        self.num_instances = len(instances)
        self.label_counts = self._count_labels()
        self.entropy = self._calculate_entropy()

    @property
    def center_signature(self) -> str:
        parts = [self.center_class_name]
        for ref, sat in self.path_conditions:
            parts.append(("+" if sat else "-") + repr(ref))
        return " & ".join(parts)
    
    def _count_labels(self) -> Dict[int, int]:
        """레이블 분포"""
        counts: Dict[int, int] = {}
        for inst in self.instances:
            label = getattr(inst, 'hasLabel', None)
            if label is not None:
                counts[label] = counts.get(label, 0) + 1
        return counts
    
    def _calculate_entropy(self) -> float:
        """엔트로피 계산"""
        if self.num_instances == 0:
            return 0.0
        
        entropy = 0.0
        for count in self.label_counts.values():
            if count > 0:
                p = count / self.num_instances
                entropy -= p * np.log2(p)
        return entropy


class SemanticDecisionTreeLearner:
    """논문 SDT 학습 알고리즘"""
    
    def __init__(
        self,
        onto,
        max_depth: int = 10,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        verbose: bool = True,
        progress_callback: Optional[Callable[[str], None]] = None,
        progress_every_nodes: int = 25,
        progress_every_candidates: int = 250,
    ):
        self.onto = onto
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.verbose = verbose

        # Optional: print-style progress reporting (wired by experiment script)
        self.progress_callback = progress_callback
        self.progress_every_nodes = max(1, int(progress_every_nodes))
        self.progress_every_candidates = max(1, int(progress_every_candidates))
        self._progress_events = 0
        
        self.refinement_generator = RefinementGenerator(onto)
        self.root = None
        self.node_counter = 0
        self.nodes = []

    def _progress(self, message: str, *, force: bool = False):
        cb = self.progress_callback
        if cb is None:
            return
        self._progress_events += 1
        if force or (self._progress_events % self.progress_every_nodes == 0):
            try:
                cb(message)
            except Exception:
                # Progress reporting must never break training.
                pass
    
    def fit(
        self,
        center_class_name: str = "Molecule",
        instances: Optional[List] = None,
    ):
        """SDT 학습"""
        logger.info(
            f"Starting SDT training with center class: {center_class_name}"
        )
        
        # Center class와 인스턴스 가져오기
        center_class = getattr(self.onto, center_class_name)
        if instances is None:
            # Backward-compatible fallback (NOT recommended for evaluation)
            instances = list(center_class.instances())
            logger.warning(
                "fit() called without instances; using ALL ontology instances. "
                "Pass train_instances to avoid leakage."
            )
        else:
            instances = list(instances)

        logger.info(f"Total instances used for training: {len(instances)}")
        self._progress(
            f"[TRAIN] start: center={center_class_name}, n={len(instances)}",
            force=True,
        )
        
        # Root 노드 생성
        self.root = SDTNode(
            center_class=center_class,
            center_class_name=center_class_name,
            instances=instances,
            center_expr=center_class,
            depth=0,
            node_id=self._get_node_id(),
            path_conditions=[],
        )
        self.nodes.append(self.root)
        
        # 재귀적 트리 구축
        self._build_tree(self.root)
        
        logger.info(f"✅ SDT training complete. Total nodes: {len(self.nodes)}")
        self._progress(
            f"[TRAIN] done: total_nodes={len(self.nodes)}",
            force=True,
        )
        
        return self.root
    
    def _get_node_id(self) -> int:
        """노드 ID 생성"""
        node_id = self.node_counter
        self.node_counter += 1
        return node_id
    
    def _build_tree(self, node: SDTNode):
        """재귀적 트리 구축"""

        self._progress(
            f"[NODE {node.node_id}] depth={node.depth} n={node.num_instances}",
        )
        
        # Stopping criteria
        if self._should_stop(node):
            self._make_leaf(node)
            return
        
        # Refinement 생성 및 최적 선택
        (
            best_refinement,
            best_gain,
            left_instances,
            right_instances,
        ) = self._find_best_refinement(node)
        
        if best_refinement is None or best_gain <= 0:
            self._make_leaf(node)
            return
        
        # Refinement 적용
        node.refinement = best_refinement

        self._progress(
            (
                f"[NODE {node.node_id}] split={best_refinement} "
                f"gain={best_gain:.4f} left={len(left_instances)} "
                f"right={len(right_instances)}"
            ),
        )
        
        if self.verbose:
            logger.info(
                f"Node {node.node_id} (depth {node.depth}): {best_refinement} "
                f"-> Left: {len(left_instances)}, "
                f"Right: {len(right_instances)}, "
                f"Gain: {best_gain:.4f}"
            )
        
        # 자식 노드 생성
        if len(left_instances) >= self.min_samples_leaf:
            left_center_expr = self._extend_center_expr(
                node.center_expr, best_refinement, satisfied=True
            )
            node.left_child = SDTNode(
                center_class=node.center_class,
                center_class_name=node.center_class_name,
                instances=left_instances,
                center_expr=left_center_expr,
                depth=node.depth + 1,
                node_id=self._get_node_id(),
                path_conditions=(
                    node.path_conditions + [(best_refinement, True)]
                ),
            )
            self.nodes.append(node.left_child)
            self._build_tree(node.left_child)
        
        if len(right_instances) >= self.min_samples_leaf:
            right_center_expr = self._extend_center_expr(
                node.center_expr, best_refinement, satisfied=False
            )
            node.right_child = SDTNode(
                center_class=node.center_class,
                center_class_name=node.center_class_name,
                instances=right_instances,
                center_expr=right_center_expr,
                depth=node.depth + 1,
                node_id=self._get_node_id(),
                path_conditions=(
                    node.path_conditions + [(best_refinement, False)]
                ),
            )
            self.nodes.append(node.right_child)
            self._build_tree(node.right_child)
        
        # 자식이 없으면 leaf
        if node.left_child is None and node.right_child is None:
            self._make_leaf(node)
    
    def _should_stop(self, node: SDTNode) -> bool:
        """Stopping criteria"""
        if node.depth >= self.max_depth:
            return True
        if node.num_instances < self.min_samples_split:
            return True
        # Pure node (single label)
        if len(node.label_counts) == 1:
            return True
        if node.entropy == 0:
            return True
        return False
    
    def _make_leaf(self, node: SDTNode):
        """Leaf 노드 생성"""
        node.is_leaf = True
        if not node.label_counts:
            node.predicted_label = 0
        else:
            node.predicted_label = max(
                node.label_counts, key=node.label_counts.get
            )
        
        if self.verbose:
            logger.info(f"Leaf {node.node_id}: Label {node.predicted_label}, "
                        f"Counts {node.label_counts}")

        self._progress(
            f"[NODE {node.node_id}] leaf label={node.predicted_label} "
            f"counts={node.label_counts}",
        )
    
    def _find_best_refinement(self, node: SDTNode) -> Tuple:
        """최적 refinement 찾기"""

        # Node-aware refinements (dynamic numeric cutpoints + cached static refinements)
        exclude = {ref for ref, _ in node.path_conditions}
        self._progress(
            f"[NODE {node.node_id}] generating candidates...",
        )

        candidates = self.refinement_generator.generate_all_refinements(
            center_signature=node.center_signature,
            instances=node.instances,
            exclude=exclude,
            enable_cache=True,
        )

        self._progress(
            f"[NODE {node.node_id}] candidates={len(candidates)}",
        )

        if len(candidates) == 0:
            return None, 0.0, [], []
        
        best_refinement = None
        best_gain = -float('inf')
        best_left = []
        best_right = []
        
        # 각 refinement의 정보 이득 계산 (single pass partition)
        for i, refinement in enumerate(candidates, start=1):
            left_instances: List = []
            right_instances: List = []

            for inst in node.instances:
                if self.refinement_generator.instance_satisfies_refinement(
                    inst, refinement
                ):
                    left_instances.append(inst)
                else:
                    right_instances.append(inst)
            
            # 유효성 + 최소 샘플 수 체크
            if (
                len(left_instances) < self.min_samples_leaf
                or len(right_instances) < self.min_samples_leaf
            ):
                continue
            
            # Information Gain 계산
            gain = self._calculate_information_gain(
                node.instances, left_instances, right_instances
            )
            
            if gain > best_gain:
                best_gain = gain
                best_refinement = refinement
                best_left = left_instances
                best_right = right_instances

            # Candidate-scan progress (guarded)
            if (
                self.progress_callback is not None
                and i % self.progress_every_candidates == 0
            ):
                self._progress(
                    f"[NODE {node.node_id}] scanned={i}/{len(candidates)} "
                    f"best_gain={best_gain:.4f}",
                )
        
        return best_refinement, best_gain, best_left, best_right

    def _extend_center_expr(
        self,
        parent_expr,
        refinement: DLRefinement,
        *,
        satisfied: bool,
    ):
        """Accumulate the path concept as a conjunction of OWL restrictions.

        This is best-effort: if a refinement cannot be converted to an Owlready2
        restriction (e.g., unsupported datatype restriction), we keep parent_expr.

        The authoritative representation for caching/logic remains
        SDTNode.path_conditions + SDTNode.center_signature.
        """

        try:
            restriction = refinement.to_owlready_restriction(self.onto)
            if restriction is None:
                return parent_expr
            if satisfied:
                return parent_expr & restriction
            return parent_expr & Not(restriction)
        except Exception:
            return parent_expr
    
    def _calculate_information_gain(
        self,
        parent_instances: List,
        left_instances: List,
        right_instances: List,
    ) -> float:
        """정보 이득 계산"""
        parent_entropy = self._calculate_entropy(parent_instances)
        
        n = len(parent_instances)
        n_left = len(left_instances)
        n_right = len(right_instances)
        
        left_entropy = self._calculate_entropy(left_instances)
        right_entropy = self._calculate_entropy(right_instances)
        
        weighted_entropy = (
            (n_left / n) * left_entropy + (n_right / n) * right_entropy
        )
        
        return parent_entropy - weighted_entropy
    
    def _calculate_entropy(self, instances: List) -> float:
        """엔트로피 계산"""
        if len(instances) == 0:
            return 0.0
        
        label_counts = {}
        for inst in instances:
            label = getattr(inst, 'hasLabel', None)
            if label is not None:
                label_counts[label] = label_counts.get(label, 0) + 1
        
        entropy = 0.0
        n = len(instances)
        for count in label_counts.values():
            if count > 0:
                p = count / n
                entropy -= p * np.log2(p)
        
        return entropy
    
    def predict(self, instance) -> int:
        """단일 인스턴스 예측"""
        node = self.root
        
        while not node.is_leaf:
            if self.refinement_generator.instance_satisfies_refinement(
                instance, node.refinement
            ):
                node = node.left_child
            else:
                node = node.right_child
            
            if node is None:
                break
        
        if node and node.is_leaf:
            return node.predicted_label
        
        return 0  # default
    
    def predict_batch(self, instances: List) -> np.ndarray:
        """여러 인스턴스 예측"""
        return np.array([self.predict(inst) for inst in instances])


if __name__ == "__main__":
    from src.ontology.ontology_loader import OntologyLoader
    
    # OWL 로드
    loader = OntologyLoader("ontology/bbbp_ontology.owl")
    onto = loader.load()
    
    # SDT 학습
    learner = SemanticDecisionTreeLearner(onto, max_depth=5, verbose=True)
    root = learner.fit("Molecule")
    
    print("\n✅ Training complete!")
    print(f"   Total nodes: {len(learner.nodes)}")
    print(f"   Max depth: {max(n.depth for n in learner.nodes)}")
