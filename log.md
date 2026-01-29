(base) PS C:\Users\Administrator\Desktop\SDT> python experiments/true_sdt_experiment.py
INFO:__main__:======================================================================
INFO:__main__:논문 SDT 실험: BBBP 데이터셋
INFO:__main__:======================================================================
INFO:__main__:
[1/5] Loading ontology...
INFO:src.ontology.ontology_loader:Loading ontology: ontology/bbbp_ontology.owl
INFO:src.ontology.ontology_loader:✅ Loaded: 2039 instances
INFO:__main__:
[2/5] Splitting data...
INFO:__main__:Total molecules: 2039
INFO:__main__:Train: 1632, Test: 407
INFO:__main__:Train labels: 0=380, 1=1252
INFO:__main__:Test labels: 0=99, 1=308
INFO:__main__:
[3/5] Training Semantic Decision Tree...
INFO:src.sdt.sdt_learner:Starting SDT training with center class: Molecule
INFO:src.sdt.sdt_learner:Total instances: 2039
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.refinement.dl_refinement_generator:Generating refinements...
INFO:src.refinement.dl_refinement_generator:  Existential: 10
INFO:src.refinement.dl_refinement_generator:  Data Property: 48
INFO:src.refinement.dl_refinement_generator:  Boolean: 4
INFO:src.refinement.dl_refinement_generator:✅ Total refinements: 62
INFO:src.sdt.sdt_learner:✅ SDT training complete. Total nodes: 147
INFO:__main__:
[4/5] Evaluating...
INFO:__main__:
[5/5] Results:
INFO:__main__:======================================================================
INFO:__main__:
✅ Performance Metrics:
INFO:__main__:   Accuracy:  0.8477
INFO:__main__:   AUC-ROC:   0.8920
INFO:__main__:
 Confusion Matrix:
INFO:__main__:
[[ 56  43]
 [ 19 289]]
INFO:__main__:
 Classification Report:
INFO:__main__:
              precision    recall  f1-score   support

    0       0.75      0.57      0.64        99
           1       0.87      0.94      0.90       308

    accuracy                           0.85       407
   macro avg       0.81      0.75      0.77       407
weighted avg       0.84      0.85      0.84       407

INFO:__main__:
📊 Tree Structure:
INFO:__main__:   Total nodes: 147
INFO:__main__:   Max depth: 8
INFO:__main__:   Leaf nodes: 74
INFO:__main__:
==========

INFO:__main__:✅ 논문 SDT 실험 완료!
INFO:__main__:======================================================================
