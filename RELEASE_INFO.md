# ADAIL Enhanced 0.4.0

Release ini diturunkan dari first commit repository ADAIL:

```text
eb244bc36dd4aec04d20dc388e44bdc72d005ddf
```

## Isi release

- source package ADAIL yang dapat diinstal dengan `pip install -e .`;
- selective expandable ResNet-18;
- alternating stability/plasticity objectives;
- cosine classifier dan weight imprinting;
- class-balanced focal, distillation, auxiliary, dan margin-ranking losses;
- balanced replay dan hybrid adaptive exemplar memory;
- differentiable prototype separation;
- checkpoint aman untuk default `weights_only=True` PyTorch modern;
- konfigurasi CCH5000, HAM10000, dan EyePACS;
- dokumentasi, examples, CLI, dan tests.

Dataset, pretrained checkpoint eksperimen, dan source LO2LN tidak disertakan.

## Verifikasi release

Pada 2026-08-19, source ini diverifikasi dengan:

```text
ruff check .                    PASS
git diff --check               PASS
pytest -q                       23 passed
examples/minimal_synthetic.py   PASS
expandable ResNet smoke test    PASS (2 branches)
```

Runtime pengujian utama menggunakan Python 3.12 dan PyTorch CPU. Untuk eksperimen medis, jalankan beberapa seed dan laporkan confidence interval; release ini tidak mengklaim peningkatan akurasi tanpa hasil eksperimen pada dataset target.

