"""
논문 SDT 전체 실험: BBBP 데이터셋
Train/Test Split + AUC-ROC 평가
"""

import sys
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent.parent))

from src.ontology.ontology_loader import OntologyLoader  # noqa: E402
from src.sdt.sdt_learner import SemanticDecisionTreeLearner  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
import numpy as np  # noqa: E402
import logging  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def p(msg: str):
    """Timestamped print for long-running progress visibility."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def split_instances(all_instances, test_ratio=0.2, random_seed=42):
    """Train/Test split"""
    np.random.seed(random_seed)
    indices = np.random.permutation(len(all_instances))
    
    test_size = int(len(all_instances) * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    train_instances = [all_instances[i] for i in train_indices]
    test_instances = [all_instances[i] for i in test_indices]
    
    return train_instances, test_instances


def get_labels(instances):
    """인스턴스에서 label 추출"""
    labels = []
    for inst in instances:
        label = getattr(inst, 'hasLabel', None)
        labels.append(label if label is not None else 0)
    return np.array(labels)


def calculate_probabilities(
    learner,
    instances,
    *,
    printer=None,
    every: int = 500,
):
    """
    트리 leaf의 label 분포로부터 확률 계산
    """
    probabilities = []
    
    for i, inst in enumerate(instances, start=1):
        if printer is not None and every > 0 and (i == 1 or i % every == 0):
            printer(f"[EVAL] probability pass: {i}/{len(instances)}")
        node = learner.root
        
        # Leaf까지 탐색
        while not node.is_leaf and node is not None:
            if learner.refinement_generator.instance_satisfies_refinement(
                inst, node.refinement
            ):
                node = node.left_child
            else:
                node = node.right_child
        
        # Leaf의 label 분포로 확률 계산
        if node and node.is_leaf:
            total = node.num_instances
            if total == 0:
                probabilities.append(0.5)
            else:
                pos_count = node.label_counts.get(1, 0)
                probabilities.append(pos_count / total)
        else:
            probabilities.append(0.5)
    
    return np.array(probabilities)


def main():
    logger.info("="*70)
    logger.info("논문 SDT 실험: BBBP 데이터셋")
    logger.info("="*70)
    p("=== SDT BBBP experiment starting ===")
    
    # 1. 온톨로지 로드
    logger.info("\n[1/5] Loading ontology...")
    p("[1/5] Loading ontology...")
    loader = OntologyLoader("ontology/bbbp_ontology.owl")
    onto = loader.load()

    # 1.5 Reasoning & semantic enrichment (required)
    p("[1/5] Running reasoner (this can take a while)...")
    loader.run_reasoner()
    p("[1/5] Reasoning complete")
    
    # 2. 전체 인스턴스 가져오기
    logger.info("\n[2/5] Splitting data...")
    p("[2/5] Splitting data...")
    all_molecules = loader.get_instances("Molecule")
    logger.info(f"Total molecules: {len(all_molecules)}")
    p(f"Total molecules: {len(all_molecules)}")
    
    train_instances, test_instances = split_instances(
        all_molecules, test_ratio=0.2
    )
    logger.info(f"Train: {len(train_instances)}, Test: {len(test_instances)}")
    p(f"Train: {len(train_instances)}, Test: {len(test_instances)}")
    
    train_labels = get_labels(train_instances)
    test_labels = get_labels(test_instances)
    logger.info(
        f"Train labels: 0={sum(train_labels==0)}, 1={sum(train_labels==1)}"
    )
    logger.info(
        f"Test labels: 0={sum(test_labels==0)}, 1={sum(test_labels==1)}"
    )
    
    # 3. SDT 학습
    logger.info("\n[3/5] Training Semantic Decision Tree...")
    p("[3/5] Training Semantic Decision Tree...")
    learner = SemanticDecisionTreeLearner(
        onto,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        verbose=False,
        progress_callback=p,
        progress_every_nodes=15,
        progress_every_candidates=500,
    )
    
    # Train-only learning (avoid leakage)
    learner.fit("Molecule", instances=train_instances)
    p("[3/5] Training finished")
    
    # 4. 예측
    logger.info("\n[4/5] Evaluating...")
    p("[4/5] Evaluating...")
    test_predictions = learner.predict_batch(test_instances)
    test_probabilities = calculate_probabilities(
        learner,
        test_instances,
        printer=p,
        every=500,
    )
    
    # 5. 평가
    logger.info("\n[5/5] Results:")
    logger.info("="*70)
    p("[5/5] Results")
    
    accuracy = accuracy_score(test_labels, test_predictions)
    auc_roc = roc_auc_score(test_labels, test_probabilities)
    
    logger.info("\n✅ Performance Metrics:")
    logger.info(f"   Accuracy:  {accuracy:.4f}")
    logger.info(f"   AUC-ROC:   {auc_roc:.4f}")
    
    logger.info("\n Confusion Matrix:")
    cm = confusion_matrix(test_labels, test_predictions)
    logger.info(f"\n{cm}")
    
    logger.info("\n Classification Report:")
    logger.info(f"\n{classification_report(test_labels, test_predictions)}")
    
    logger.info("\n📊 Tree Structure:")
    logger.info(f"   Total nodes: {len(learner.nodes)}")
    logger.info(f"   Max depth: {max(n.depth for n in learner.nodes)}")
    logger.info(f"   Leaf nodes: {sum(1 for n in learner.nodes if n.is_leaf)}")
    
    logger.info("\n" + "="*70)
    logger.info("✅ 논문 SDT 실험 완료!")
    logger.info("="*70)
    p("=== SDT BBBP experiment done ===")


if __name__ == "__main__":
    main()
