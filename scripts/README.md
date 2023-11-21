## Experiment scripts for our methods and baselines
-----------------------

### Zero-shot comparisons

run_pt_eval.sh
waffle_descriptors_eval.sh

### Few-shot OOD comparisons

| Method | Command to run |
| ------ | -------------- |
| CLIP-adapter | `scripts/run_adapter.sh 6e-3 ViT-B-16 512` |
| bitfit | `scripts/bitfit.sh 1.25e-4 ViT-B-16 512` |
| Cross Entropy | `scripts/run_ce.sh 2e-5 ViT-B-16 512` |
| Cross Entropy + word soup + diversity loss | `scripts/run_ce_regularized.sh 0.25 10` |
| ClipOOD | `scripts/run_clipood.sh 2e-5 ViT-B-16 512` |
| ClipOOD + word soup + diversity loss | `scripts/run_clipood_regularized.sh 0.25 10` |
| CoOp | `scripts/run_coop.sh 8e-5 ViT-B-16 512` |
| CoOp + word soup + diversity loss | `scripts/run_coop_regularized.sh 0.25 10` |
| KgCoOp |  `scripts/run_kgcoop.sh 4e-5 ViT-B-16 512` |
| LoRA |  `scripts/run_lora.sh 1e-5 ViT-B-16 512` |
| MaPLe |  `scripts/run_maple.sh 0.025 ViT-B-16 512` |
| MaPLe + word soup + diversity loss |  `scripts/run_maple_regularized.sh` |
| ProDA |  `scripts/run_proda.sh 3.2e-4 ViT-B-16 512` |
| ProGrad |  `scripts/run_prograd.sh 1.28e-3 ViT-B-16 512` |
| ResBlock-adapter | `scripts/run_resblock_adapter.sh 2.5e-3 ViT-B-16 512` |
| SSF | `scripts/run_ssf.sh 1e-4 ViT-B-16 512` |
| VPT | `scripts/run_vpt_deep.sh 0.8 ViT-B-16 512` |

### Other scripts

run_ce_with_eval.btn.sh
run_ce_with_eval.sh
run_clipood_with_eval.btn.sh
run_clipood_with_eval.sh
run_coop_with_eval.btn.sh
run_coop_with_eval.sh
run_maple_with_eval.btn.sh
run_maple_with_eval.sh
