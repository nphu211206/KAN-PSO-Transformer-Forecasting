# Environment Manifest

## Canonical training environment

Recorded identically in the canonical JSON evidence:

- Python 3.12.13
- PyTorch 2.10.0+cu128
- CUDA runtime 12.8
- NumPy 2.0.2
- pandas 2.3.3
- scikit-learn 1.6.1
- transformers 5.0.0
- GPU: NVIDIA Tesla T4

These versions describe the Kaggle training environment. They are evidence,
not a requirement for reading or auditing frozen JSON.

## Publication environment

`audit_evidence.py`, `audit_manuscript.py` and `publication_tables.py` run with
the Python standard library. `analysis.py` additionally requires pandas,
NumPy and matplotlib when figures must be regenerated.

The legacy `.venv` points to a no-longer-accessible Python 3.13 installation
and must not be presented as the canonical training environment.
