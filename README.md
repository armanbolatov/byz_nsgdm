# Byz-NSGDM: Byzantine-Robust Normalized SGD with Momentum

This repository contains the code for our paper on Byzantine-robust optimization under $(L_0, L_1)$-smoothness. We propose **Byz-NSGDM**, which combines gradient normalization with momentum and robust aggregation to achieve convergence guarantees under generalized smoothness conditions.

This is a fork of [epfml/byzantine-robust-noniid-optimizer](https://github.com/epfml/byzantine-robust-noniid-optimizer). We are grateful to **Sai Praneeth Karimireddy**, **Lie He**, and **Martin Jaggi** from the [EPFML lab](https://www.epfl.ch/labs/mlo/) for their clean and reproducible codebase.

## Code organization

- `codes/` — Core framework (workers, server, simulator, aggregators, attacks, samplers)
- `exp{1-10}.py` — Launcher scripts for MNIST experiments
- `exp11/` — Synthetic quartic optimization ($f(x) = \|x\|^4$)
- `exp12/` — Character-level language modeling (Shakespeare, small GPT)
- `outputs/` — Experiment results

## Reproduction

1. Install dependencies: `pip install -r requirements.txt`
2. Run experiments:
```bash
bash run.sh              # MNIST experiments (interactive menu)
bash run_ablation.sh     # Momentum/LR ablation study
cd exp11 && python main.py   # Synthetic quartic optimization
cd exp12 && bash run_exp12.sh   # Shakespeare LM
```
3. Results are saved to the corresponding folders under `outputs/`

## License

This repo is covered under [The MIT License](LICENSE).

## Acknowledgments

This codebase builds on the work of Karimireddy et al.:

```
@inproceedings{karimireddy2022byzantinerobust,
      title={Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing},
      author={Sai Praneeth Karimireddy and Lie He and Martin Jaggi},
      booktitle={International Conference on Machine Learning (ICML)},
      year={2022}
}
```
