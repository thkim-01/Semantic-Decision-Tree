"""src.ontology.ontology_loader

OWL 온톨로지 로더 + HermiT / Pellet Reasoner 통합.

실험/학습 실행 시 Owlready2 reasoner가 매우 많은 "Reparenting ..." 로그를
stdout/stderr로 출력할 수 있어, 기본적으로는 해당 출력을 억제한다.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import List

from owlready2 import (
    get_ontology,
    set_log_level,
    sync_reasoner,
    sync_reasoner_hermit,
    sync_reasoner_pellet,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OntologyLoader:
    """OWL 파일 로드 + Reasoner 실행"""
    
    def __init__(self, owl_path: str):
        self.owl_path = owl_path
        self.onto = None
        self.reasoner_synced = False
        
    def load(self):
        """온톨로지 로드"""
        logger.info(f"Loading ontology: {self.owl_path}")
        self.onto = get_ontology(self.owl_path).load()
        logger.info(
            f"✅ Loaded: {len(list(self.onto.individuals()))} instances"
        )
        return self.onto

    def save(self, owl_path: str):
        """Save the currently loaded ontology to a file."""

        if self.onto is None:
            raise ValueError("Ontology not loaded!")
        self.onto.save(file=owl_path)
    
    @staticmethod
    @contextlib.contextmanager
    def _suppress_output(enabled: bool):
        if not enabled:
            yield
            return
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(
                devnull
            ), contextlib.redirect_stderr(devnull):
                yield

    def run_reasoner(
        self,
        reasoner: str = "HermiT",
        *,
        suppress_output: bool = True,
    ):
        """Run a DL reasoner (HermiT / Pellet) to enrich class memberships.

        Args:
            reasoner: "HermiT" (default), "Pellet", or "sync_reasoner".
            suppress_output: If True, suppresses noisy Owlready2 stdout/stderr.
        """
        if self.onto is None:
            raise ValueError("Ontology not loaded!")
        
        logger.info(f"Running {reasoner} reasoner...")

        # Reduce Owlready internal logging; this does not always cover all
        # prints, so we also optionally redirect stdout/stderr.
        if suppress_output:
            try:
                set_log_level(0)
            except Exception:
                pass

        with self._suppress_output(suppress_output):
            with self.onto:
                try:
                    if reasoner.lower() == "hermit":
                        sync_reasoner_hermit(infer_property_values=True)
                    elif reasoner.lower() == "pellet":
                        sync_reasoner_pellet(infer_property_values=True)
                    else:
                        sync_reasoner(infer_property_values=True)

                    self.reasoner_synced = True
                    logger.info("✅ Reasoning complete!")
                except Exception as e:
                    logger.warning(
                        f"Reasoner failed: {e}. Continuing without reasoning."
                    )
                    self.reasoner_synced = False
    
    def get_instances(self, class_name: str) -> List:
        """특정 클래스의 모든 인스턴스 반환"""
        cls = getattr(self.onto, class_name, None)
        if cls is None:
            return []
        return list(cls.instances())
    
    def get_instance_labels(self, instances: List) -> List[int]:
        """인스턴스들의 label 추출"""
        labels = []
        for inst in instances:
            label_value = getattr(inst, 'hasLabel', None)
            if label_value is not None:
                labels.append(label_value)
            else:
                labels.append(None)
        return labels
    
    def get_all_classes(self) -> List:
        """모든 클래스 반환"""
        return list(self.onto.classes())
    
    def get_object_properties(self) -> List:
        """모든 Object Property 반환"""
        return list(self.onto.object_properties())
    
    def get_data_properties(self) -> List:
        """모든 Data Property 반환"""
        return list(self.onto.data_properties())
    
    def get_functional_group_classes(self) -> List:
        """FunctionalGroup의 모든 서브클래스 반환"""
        fg_class = getattr(self.onto, 'FunctionalGroup', None)
        if fg_class is None:
            return []
        return list(fg_class.descendants())


if __name__ == "__main__":
    # 테스트
    loader = OntologyLoader("ontology/bbbp_ontology.owl")
    onto = loader.load()
    loader.run_reasoner()
    
    molecules = loader.get_instances("Molecule")
    labels = loader.get_instance_labels(molecules)
    
    print(f"\nMolecules: {len(molecules)}")
    print(f"Labels: {labels[:10]}")
    print(f"Label distribution: 0={labels.count(0)}, 1={labels.count(1)}")
