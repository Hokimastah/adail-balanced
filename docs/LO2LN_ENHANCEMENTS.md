# Peningkatan ADAIL 0.4.0 yang Terinspirasi LO2LN

Dokumen ini menjelaskan perubahan teknis dari ADAIL 0.3.0. Implementasi dibuat khusus untuk ADAIL dan tidak menyalin source code LO2LN. Rujukan konseptual utama adalah paper [LO2LN: Learning from the Old and Learning from the New](https://arxiv.org/abs/2303.13752) dan repository baseline yang sudah didukung oleh adapter proyek.

## Ringkasan perubahan

| Area | Implementasi 0.4.0 | Tujuan |
|---|---|---|
| Representasi | Ekspansi selektif `layer4` ResNet-18 + gated residual | Menambah kapasitas untuk kelas baru tanpa mengubah dimensi fitur |
| Optimisasi | Fase `new` dan `old` bergantian | Menyeimbangkan plasticity dan stability |
| Classifier | Cosine classifier dengan learned scale | Mengurangi bias norm bobot terhadap kelas baru |
| Inisialisasi | Imprinting bobot dari centroid fitur | Memberi kelas baru titik awal yang bermakna |
| Imbalance | Class-balanced focal loss | Menekan dominasi kelas mudah/mayoritas |
| Retensi | Logit distillation pada kelas lama | Menjaga respons teacher task sebelumnya |
| Batas kelas | Old-vs-new margin ranking | Menjaga skor kelas lama di atas kelas baru pada exemplar lama |
| Auxiliary | Loss pada fitur cabang terbaru | Memastikan cabang baru benar-benar diskriminatif |
| Replay | Batch class-balanced dengan proporsi memory tetap | Menjamin exemplar lama hadir secara konsisten |
| Memory | Adaptive quota + herding + uncertainty + diversity | Menyimpan contoh yang representatif sekaligus informatif |
| Prototype | Separation loss yang differentiable | Membuat separation benar-benar memberi gradien ke backbone |
| Checkpoint | Container aman untuk `weights_only=True` | Kompatibel dengan default PyTorch 2.6+ |

## Arsitektur selective expansion

Backbone baru membagi ResNet-18 menjadi:

- shared trunk: stem, `layer1`, `layer2`, dan `layer3`;
- base branch: `layer4` task pertama;
- incremental branch: salinan `layer4` terbaru pada setiap task berikutnya;
- projection: transformasi identitas trainable 512 ke 512;
- residual gate: skalar sigmoid yang dimulai kecil (`dynamic_gate_init=-2`).

Fitur gabungan task ke-`t` adalah:

```text
z_t = z_base + sum(sigmoid(g_i) * P_i(z_i))
```

Dimensi tetap 512 menjaga classifier, drift aligner, dan prototype bank tetap kompatibel. Cabang lama dan shared trunk dibekukan setelah base task; BatchNorm pada komponen beku selalu berada dalam mode evaluasi.

## Objective per fase

Fase `new` mengaktifkan cabang terbaru dan auxiliary classification. Fase `old` membekukan cabang terbaru, men-zero-kan gradien row classifier kelas baru, serta memperbesar bobot distillation dan ranking. Jadwal default adalah satu fase stability setiap dua epoch.

Total objective:

```text
L = L_cb_focal
  + lambda_aux   L_aux
  + lambda_kd    m_kd(phase) L_distill
  + lambda_rank  m_rank(phase) L_ranking
  + lambda_drift L_drift
  + lambda_proto L_compact
  + lambda_sep   L_sep_diff
  + lambda_ord   L_ordinal
```

Distillation secara default hanya diterapkan ke exemplar memory. Margin ranking juga hanya aktif ketika batch mengandung target lama dan task memiliki kelas baru. Implementasi ranking menggunakan tensor dua dimensi yang konsisten, sehingga tidak terkena error `MarginRankingLoss` akibat target satu dimensi pada PyTorch modern.

## Balanced replay

`ReplayDataset` mengonversi setiap label menjadi scalar `torch.long`. Ini mencegah error collate ketika dataset baru mengembalikan `int` atau array `(1,)`, sementara memory mengembalikan tensor. `BalancedReplayBatchSampler` kemudian memilih kelas secara merata dan mempertahankan `replay_memory_fraction` per batch.

## Hybrid memory

Kuota awal per kelas tetap memakai uncertainty, forgetting, dan rarity milik ADAIL. Kandidat per kelas dipilih dalam tiga tahap:

1. herding untuk mendekati centroid kelas;
2. entropy uncertainty untuk kasus informatif;
3. greedy k-center untuk diversity.

Jika suatu kelas tidak memiliki cukup kandidat untuk memenuhi kuota, sisa kapasitas dibagikan secara deterministik ke kelas yang masih memiliki kandidat. State memory diserialisasi sebagai tensor, angka, list, dan dictionary saja.

## Prototype separation yang benar-benar trainable

Prototype EMA adalah buffer dan tidak mempunyai gradien. Karena itu, menghitung jarak hanya antarbuffers menghasilkan nilai diagnostik tetapi tidak mengubah model. Versi 0.4.0 menghitung centroid dari fitur batch aktif dan membandingkannya dengan prototype pesaing yang didetach. Gradien mengalir ke fitur/backbone, sedangkan prototype tetap stabil.

## Ablation yang disarankan

Gunakan seed, class order, augmentasi, memory size, epoch, dan split data yang sama. Ubah satu kelompok opsi pada satu waktu:

1. baseline 0.3-like: matikan expansion, alternating, focal, auxiliary, distillation, ranking, dan balanced replay;
2. tambah cosine classifier + imprinting;
3. tambah balanced focal + balanced replay;
4. tambah distillation + ranking;
5. tambah selective expansion + alternating phases;
6. bandingkan `memory_selection: random`, `herding`, dan `hybrid`;
7. bandingkan separation buffer-only (diagnostik) dengan `L_sep_diff`.

Laporkan minimal average incremental accuracy, average forgetting, per-class accuracy, dan—untuk label ordinal—MAE serta QWK. Perubahan ini adalah hipotesis engineering; peningkatan akurasi harus dibuktikan melalui eksperimen berulang dan confidence interval, bukan diasumsikan dari desain.

## Kompatibilitas

- Python 3.10–3.12 dan PyTorch 2.x.
- Config 0.3.0 tetap dapat dibaca karena field baru mempunyai default.
- `ADAILNet.from_config(config)` adalah cara yang disarankan agar opsi arsitektur YAML tidak terlewat.
- Checkpoint 0.4.0 dimuat dengan `weights_only=True` bila API tersedia.
- Checkpoint 0.3.0 dengan objek `adail.memory.Exemplar` didukung melalui safe-global terbatas untuk kelas tersebut.

