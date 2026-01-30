"""
논문 SDT 전체 실험: BBBP 데이터셋
Train/Test Split + AUC-ROC 평가
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

sys.path.append(str(Path(__file__).parent.parent))

from src.ontology.ontology_loader import OntologyLoader  # noqa: E402
from src.sdt.sdt_learner import SemanticDecisionTreeLearner  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def p(msg: str):
    """Timestamped print for long-running progress visibility."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_ontology_with_inference_cache(
    *,
    base_owl: Path,
    inferred_owl: Path,
    cache_meta: Path,
    force_reasoner: bool,
    reasoner: str,
):
    """Load ontology, reusing a cached inferred OWL when valid.

    Validity is checked by comparing the base OWL sha256 recorded in metadata.
    If ontology content changes (TBox/ABox), the hash changes and we rerun.

    Note: if semantics change without changing the OWL file (e.g., external
    SWRL rules injected at runtime), use --force-reasoner.
    """

    base_owl = base_owl.resolve()
    inferred_owl = inferred_owl.resolve()
    cache_meta = cache_meta.resolve()

    base_hash = sha256_file(base_owl)

    if not force_reasoner and inferred_owl.exists() and cache_meta.exists():
        try:
            meta = json.loads(cache_meta.read_text(encoding="utf-8"))
            if meta.get("base_sha256") == base_hash and meta.get(
                "reasoner"
            ) == reasoner:
                p(
                    "[1/5] Using cached inferred ontology: "
                    f"{inferred_owl.name}"
                )
                loader = OntologyLoader(str(inferred_owl))
                onto = loader.load()
                return onto, loader, {
                    "reasoner_ran": False,
                    "base_sha256": base_hash,
                    "inferred_owl": str(inferred_owl),
                }
        except Exception:
            # If metadata is corrupt, fall back to rebuilding.
            pass

    p("[1/5] Building inferred ontology (reasoner will run once)...")
    loader = OntologyLoader(str(base_owl))
    onto = loader.load()

    t0 = time.perf_counter()
    loader.run_reasoner(suppress_output=True, reasoner=reasoner)
    reasoner_seconds = time.perf_counter() - t0

    inferred_owl.parent.mkdir(parents=True, exist_ok=True)
    onto.save(file=str(inferred_owl))

    cache_payload = {
        "dataset": "bbbp",
        "base_owl": str(base_owl),
        "base_sha256": base_hash,
        "reasoner": reasoner,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inferred_owl": str(inferred_owl),
        "reasoner_seconds": reasoner_seconds,
    }
    cache_meta.write_text(
        json.dumps(cache_payload, indent=2),
        encoding="utf-8",
    )
    p(f"[1/5] Saved inferred ontology: {inferred_owl}")

    return onto, loader, {
        "reasoner_ran": True,
        "reasoner_seconds": reasoner_seconds,
        "base_sha256": base_hash,
        "inferred_owl": str(inferred_owl),
    }


def export_artifacts(
    *,
    out_dir: Path,
    dataset_name: str,
    run_started_at: datetime,
    base_owl: str,
    inferred_owl: str,
    reasoner_info: dict,
    hyperparams: dict,
    split_info: dict,
    test_labels: np.ndarray,
    test_predictions: np.ndarray,
    test_probabilities: np.ndarray,
    cm: np.ndarray,
    learner,
):
    # Ensure plotting works in headless environments.
    try:
        plt.switch_backend("Agg")
    except Exception:
        pass

    out_dir.mkdir(parents=True, exist_ok=True)

    # confusion_matrix.png
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5), dpi=150)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax_cm, values_format="d", colorbar=False)
    ax_cm.set_title(f"{dataset_name.upper()} Confusion Matrix")
    fig_cm.tight_layout()
    fig_cm.savefig(out_dir / "confusion_matrix.png")
    plt.close(fig_cm)

    # roc_curve.png
    fpr, tpr, _ = roc_curve(test_labels, test_probabilities)
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5), dpi=150)
    ax_roc.plot(fpr, tpr, label="ROC")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"{dataset_name.upper()} ROC Curve")
    ax_roc.legend(loc="lower right")
    fig_roc.tight_layout()
    fig_roc.savefig(out_dir / "roc_curve.png")
    plt.close(fig_roc)

    # feature_importance.png (tree usage frequency, SDT-friendly)
    # We treat each refinement (or property) as a "feature" and count how
    # frequently it appears in internal nodes.
    feature_counts: Counter[str] = Counter()
    for n in getattr(learner, "nodes", []):
        ref = getattr(n, "refinement", None)
        if ref is None:
            continue
        rtype = getattr(ref, "type", "unknown")
        if rtype == "numeric":
            key = getattr(ref, "property", "numeric")
        elif rtype == "existential":
            key = (
                f"∃{getattr(ref, 'property', '?')}."
                f"{getattr(ref, 'target', '?')}"
            )
        elif rtype == "isa":
            key = f"isa({getattr(ref, 'target', '?')})"
        elif rtype == "value":
            key = (
                f"{getattr(ref, 'property', '?')}="
                f"{getattr(ref, 'value', '?')}"
            )
        else:
            key = str(ref)
        feature_counts[key] += 1

    fig_fi, ax_fi = plt.subplots(figsize=(10, 6), dpi=150)
    top = feature_counts.most_common(15)
    labels = [k for k, _ in top][::-1]
    vals = [v for _, v in top][::-1]
    ax_fi.barh(labels, vals)
    ax_fi.set_xlabel("Usage count in internal nodes")
    ax_fi.set_title(
        f"{dataset_name.upper()} Feature Importance (usage frequency)"
    )
    fig_fi.tight_layout()
    fig_fi.savefig(out_dir / "feature_importance.png")
    plt.close(fig_fi)

    # metrics.txt (compatible with existing example format + extra metadata)
    accuracy = accuracy_score(test_labels, test_predictions)
    auc_roc = roc_auc_score(test_labels, test_probabilities)
    report = classification_report(test_labels, test_predictions)

    metrics_path = out_dir / "metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write(f"{dataset_name.upper()} Experiment Results\n")
        f.write("=" * 50 + "\n\n")

        f.write(
            "Run started at: "
            f"{run_started_at.isoformat(timespec='seconds')}\n"
        )
        f.write(f"Base ontology: {base_owl}\n")
        f.write(f"Inferred ontology: {inferred_owl}\n")
        f.write(
            f"Reasoner ran: {reasoner_info.get('reasoner_ran', False)}\n"
        )
        if reasoner_info.get("reasoner_ran"):
            seconds = float(reasoner_info.get("reasoner_seconds", 0.0))
            f.write(f"Reasoner seconds: {seconds:.2f}\n")
        f.write(f"Base sha256: {reasoner_info.get('base_sha256', '')}\n")
        f.write("\n")

        f.write("Split info:\n")
        for k, v in split_info.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write("Hyperparameters:\n")
        for k, v in hyperparams.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"AUC-ROC:   {auc_roc:.4f}\n\n")

        f.write("Confusion Matrix:\n")
        f.write(f"{cm}\n\n")

        f.write("Classification Report:\n")
        f.write(report + "\n")

        f.write("\nTree Structure:\n")
        nodes = getattr(learner, "nodes", [])
        f.write(f"Total nodes: {len(nodes)}\n")
        if nodes:
            f.write(f"Max depth: {max(n.depth for n in nodes)}\n")

        f.write("\nFeature Importance (usage frequency, top 15):\n")
        total = sum(feature_counts.values()) or 1
        for k, v in feature_counts.most_common(15):
            f.write(f"  {k:35s}: {v / total:.4f} ({v})\n")


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
    
    args = build_arg_parser().parse_args()
    run_started_at = datetime.now()

    dataset_name = "bbbp"
    base_owl = Path(args.base_owl)
    inferred_owl = Path(args.inferred_owl)
    cache_meta = Path(args.inferred_meta)

    # 1. 온톨로지 로드 (cached inferred ontology if available)
    logger.info("\n[1/5] Loading ontology...")
    p("[1/5] Loading ontology...")
    onto, loader, reasoner_info = load_ontology_with_inference_cache(
        base_owl=base_owl,
        inferred_owl=inferred_owl,
        cache_meta=cache_meta,
        force_reasoner=args.force_reasoner,
        reasoner=args.reasoner,
    )
    
    # 2. 전체 인스턴스 가져오기
    logger.info("\n[2/5] Splitting data...")
    p("[2/5] Splitting data...")
    all_molecules = loader.get_instances("Molecule")
    logger.info(f"Total molecules: {len(all_molecules)}")
    p(f"Total molecules: {len(all_molecules)}")
    
    train_instances, test_instances = split_instances(
        all_molecules, test_ratio=args.test_ratio, random_seed=args.seed
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
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        verbose=False,
        progress_callback=p,
        progress_every_nodes=15,
        progress_every_candidates=500,
    )
    
    # Train-only learning (avoid leakage)
    learner.fit("Molecule", instances=train_instances)
    p("[3/5] Training finished")

    # Sanity-check: did we actually use any reasoner-driven isa(Class) splits?
    split_types = Counter(
        getattr(n.refinement, "type", "unknown")
        for n in learner.nodes
        if getattr(n, "refinement", None) is not None
    )
    if split_types:
        p(f"[3/5] Split type counts: {dict(split_types)}")
        p(f"[3/5] isa splits used: {split_types.get('isa', 0)}")
    
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

    # --- Output artifacts (dated + performance in folder name) ---
    out_root = Path("output")
    run_tag = run_started_at.strftime("%Y%m%d_%H%M%S")
    tmp_out_dir = out_root / f"{dataset_name}_results_{run_tag}"

    split_info = {
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "train_n": len(train_instances),
        "test_n": len(test_instances),
    }
    hyperparams = {
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
    }

    export_artifacts(
        out_dir=tmp_out_dir,
        dataset_name=dataset_name,
        run_started_at=run_started_at,
        base_owl=str(base_owl),
        inferred_owl=str(inferred_owl),
        reasoner_info=reasoner_info,
        hyperparams=hyperparams,
        split_info=split_info,
        test_labels=test_labels,
        test_predictions=test_predictions,
        test_probabilities=test_probabilities,
        cm=cm,
        learner=learner,
    )

    # Rename folder to include performance (date + AUC-ROC)
    final_out_dir = out_root / (
        f"{dataset_name}_results_{run_tag}_auc{auc_roc:.4f}"
    )
    try:
        if final_out_dir.exists():
            # Avoid collisions.
            final_out_dir = out_root / (
                f"{dataset_name}_results_{run_tag}_auc{auc_roc:.4f}_1"
            )
        tmp_out_dir.rename(final_out_dir)
        p(f"[OUTPUT] Saved artifacts to: {final_out_dir}")
    except Exception:
        p(f"[OUTPUT] Saved artifacts to: {tmp_out_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="True SDT experiment (BBBP) with inference cache + outputs"
    )
    parser.add_argument(
        "--base-owl",
        default="ontology/bbbp_ontology.owl",
        help="Base ontology OWL (pre-reasoning)",
    )
    parser.add_argument(
        "--inferred-owl",
        default="ontology/bbbp_inferred.owl",
        help="Cached inferred ontology OWL (post-reasoning)",
    )
    parser.add_argument(
        "--inferred-meta",
        default="ontology/bbbp_inferred.meta.json",
        help="Metadata JSON for inferred cache validation",
    )
    parser.add_argument(
        "--reasoner",
        default="HermiT",
        help="Reasoner to use when rebuilding inferred ontology (HermiT/Pellet)",
    )
    parser.add_argument(
        "--force-reasoner",
        action="store_true",
        help="Force reasoner run even if inferred cache exists",
    )

    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-split", type=int, default=20)
    parser.add_argument("--min-samples-leaf", type=int, default=10)

    return parser


if __name__ == "__main__":
    main()
