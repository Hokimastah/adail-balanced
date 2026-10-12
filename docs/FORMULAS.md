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
quota_c = floor(M_t * score_c / sum_j score_j)
```

Keterangan:

- `U_c`: rata-rata entropy prediksi kelas `c`.
- `F_c`: penurunan akurasi kelas `c` dari nilai terbaik historisnya.
- `R_c`: nilai kelangkaan kelas, dihitung terbalik dari jumlah sampel kelas yang telah diamati.
- `M_t`: kapasitas memori setelah task `t`.
  - `memory_budget="global"`: `M_t = memory_size` sejak task pertama.
  - `memory_budget="per_seen_class"`: `M_t = (memory_size // num_classes) * n_terlihat`, sama dengan LO2LN/iCaRL `fixed_memory`.

Bila `train_eval_dataset` diberikan, entropy dan fitur untuk seleksi exemplar dihitung dari citra tanpa augmentasi (transform evaluasi), seperti herding LO2LN.

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
L_ranking = mean max(0, cos_new_topk - cos_true_old + margin)
```

Sejak 0.5.0 skor yang dibandingkan adalah similarity cosine tanpa skala (`ranking_space="cosine"`), sama seperti LO2LN. Versi 0.4.0 memakai logit yang sudah dikali skala `s ≈ 10`, sehingga margin 0.2 efektif hanya 0.02 dalam ruang cosine. Perilaku lama tersedia lewat `ranking_space="logit"`.

Pada dataset ordinal, margin diperkecil untuk kelas klinis yang berdekatan dan diperbesar untuk kelas yang jauh.

Pada fase `old`, baris classifier kelas baru tidak ikut diperbarui sama sekali. Nilai dan momentum baris itu dipulihkan setelah `optimizer.step()`; di 0.4.0 gradiennya hanya dinolkan, sehingga momentum dan weight decay tetap menggesernya.

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

## 12. Balanced Replay

Setiap batch berukuran `B` dibagi antara exemplar dan data task berjalan:

```text
fixed:         B_mem = round(B * replay_memory_fraction)
proportional:  B_mem = round(B * n_lama / (n_lama + n_baru))
B_cur = B - B_mem
```

Di dalam tiap bagian, kelas diputar merata. Dengan `fixed` (0.5), kelas baru selalu mengisi separuh batch, sehingga tampil `n_lama` kali lebih sering daripada tiap kelas lama. Dengan `proportional`, setiap kelas terlihat mendapat jatah yang kira-kira sama.

Panjang epoch:

```text
current: ceil(n_current / B_cur)              (0.4.0)
pool:    ceil((n_current + n_memory) / B)
```

`replay_reaugment=True` menyimpan exemplar sebagai indeks dataset dan memuat ulang citranya melalui transform training setiap kali diputar ulang.

## 13. Evaluasi

Metrik di `history`:

- `average_incremental_accuracy`: rata-rata atas task dari akurasi pada seluruh sampel test kelas yang terlihat (berbobot sampel; definisi iCaRL/LO2LN).
- `average_accuracy`: rata-rata atas task dari rata-rata *tanpa bobot* akurasi per blok task. Bukan angka yang dilaporkan LO2LN.
- `average_forgetting` dan per-class forgetting estimate.
- `seen_balanced_accuracy` per task (macro recall atas kelas terlihat).
- `old_to_new_confusion` per task dan `average_old_to_new_confusion`: persentase sampel test kelas lama yang diprediksi sebagai kelas baru task itu.
- MAE dan Quadratic Weighted Kappa (QWK).
