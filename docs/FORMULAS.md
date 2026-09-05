# ADAIL Formulas

Dokumen ini merangkum komponen matematis yang diimplementasikan dari proposal **Adaptive Drift-Aware Incremental Learning (ADAIL) untuk Klasifikasi Citra Medis**.

## 1. Drift-Aware Feature Alignment

Pada task inkremental `t`, model aktif menghasilkan fitur `f_t(x)`, sedangkan frozen teacher dari task sebelumnya menghasilkan `f_{t-1}(x)`. ADAIL menggunakan fungsi penyelarasan learnable `A_t`:

```text
L_drift = || normalize(f_t(x)) - normalize(A_t(f_{t-1}(x))) ||_2^2
```

Implementasi: `src/adail/modules/drift.py`.

## 2. Adaptive Memory Allocation

Kuota exemplar kelas `c` dihitung dari tiga sinyal proposal: uncertainty, forgetting rate, dan class rarity.

```text
score_c = alpha * U_c + beta * F_c + gamma * R_c
quota_c = floor(memory_size * score_c / sum_j score_j)
```

Keterangan:

- `U_c`: rata-rata entropy prediksi kelas `c`.
- `F_c`: penurunan akurasi kelas `c` dari nilai terbaik historisnya.
- `R_c`: nilai kelangkaan kelas, dihitung terbalik dari jumlah sampel kelas yang telah diamati.

Implementasi: `src/adail/memory.py` dan `ADAILTrainer.update_memory()`.

## 3. Prototype Evolution

Prototipe kelas `p_c` diperbarui dengan exponential moving average agar stabil, namun tetap adaptif terhadap distribusi fitur task baru.

```text
p_c <- m * p_c + (1 - m) * mean(f_i | y_i = c)
```

Implementasi: `src/adail/modules/prototype.py`.

## 4. Prototype Compactness dan Differentiable Separation

Compactness menjaga fitur dekat ke prototipe kelasnya.

```text
L_proto = mean || normalize(f_i) - normalize(p_{y_i}) ||_2^2
```

Separation versi trainable memakai centroid fitur batch aktif `b_c` sebagai sumber dan prototype EMA kelas pesaing sebagai target stabil.

```text
L_sep_diff = mean_{c != j} max(0, margin(c,j)
             - || normalize(b_c) - stopgrad(normalize(p_j)) ||_2)^2
```

`PrototypeBank.separation_loss()` tetap tersedia sebagai diagnostik buffer-only. Optimisasi memakai `PrototypeBank.differentiable_separation_loss()` agar gradien mengalir ke fitur.

## 5. Cosine Classification dan Weight Imprinting

Classifier menormalisasi fitur dan row bobot sebelum menghitung logit:

```text
logit_c = s * dot(normalize(f), normalize(w_c))
w_c(new) <- normalize(mean(f_i | y_i = c))
```

`s` dapat dipelajari dan dibatasi pada rentang numerik aman. Row kelas baru diinisialisasi dari centroid fitur task baru.

## 6. Class-Balanced Focal Loss

Untuk jumlah sampel tersedia `n_c`, bobot effective-number adalah:

```text
alpha_c = (1 - beta) / (1 - beta^n_c)
L_cb_focal = mean -alpha_y * (1 - p_y)^gamma * log(p_y)
```

Bobot kelas aktif dinormalisasi agar rata-ratanya satu.

## 7. Logit Distillation

Teacher adalah snapshot model sebelum ekspansi task berjalan. Distillation hanya membandingkan kelas lama `C_old` dan secara default hanya exemplar memory:

```text
L_distill = T^2 * KL(
    softmax(z_teacher[C_old] / T)
    || softmax(z_student[C_old] / T)
)
```

## 8. Old-vs-New Margin Ranking

Untuk exemplar lama, skor target lama dijaga di atas `top-k` skor kelas baru:

```text
L_ranking = mean max(0, z_new_topk - z_true_old + margin)
```

Pada dataset ordinal, margin diperkecil untuk kelas klinis yang berdekatan dan diperbesar untuk kelas yang jauh.

## 9. Selective Expansion dan Auxiliary Objective

Cabang `layer4` baru menghasilkan fitur `z_i`, diproyeksikan oleh `P_i`, dan difusikan dengan gate `g_i`:

```text
f_t = f_base + sum_i sigmoid(g_i) * P_i(z_i)
L_aux = classification_loss(classifier(P_latest(z_latest)), y)
```

Output tetap 512-dimensional meskipun jumlah cabang bertambah.

## 10. Ordinal Constraint Learning

Untuk dataset berurutan secara klinis seperti EyePACS, ADAIL memakai expected ordinal prediction:

```text
pred = sum_k softmax(logits)_k * k
L_ordinal = mean | pred - y |
```

Implementasi: `src/adail/modules/ordinal.py`.

## 11. Total Objective

```text
L_total = L_cb_focal
        + lambda_aux     * L_aux
        + lambda_distill * phase_multiplier_kd   * L_distill
        + lambda_ranking * phase_multiplier_rank * L_ranking
        + lambda_drift   * L_drift
        + lambda_proto   * L_proto
        + lambda_sep     * L_sep_diff
        + lambda_ordinal * L_ordinal
```

## 12. Evaluasi

Metrik yang tersedia:

- Average Accuracy
- Average Forgetting
- per-class forgetting estimate
- MAE
- Quadratic Weighted Kappa (QWK)
