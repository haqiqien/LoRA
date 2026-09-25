# Laporan Eksperimen LoRA

Notebook yang digunakan adalah [`section4.ipynb`](section4.ipynb), versi hasil perbaikan [`LoRA-default.ipynb`](LoRA-default.ipynb), dan notebook eksperimen [`LoRA-exp.ipynb`](LoRA-exp.ipynb).

### Identitas dan peran file

| File | Nama yang digunakan dalam laporan | Peran |
|---|---|---|
| [`section4.ipynb`](section4.ipynb) | **Notebook Original Google Classroom** | File awal dari Google Classroom. File ini menjadi dasar pengerjaan, tetapi mengalami beberapa error saat dijalankan di Google Colab. |
| [`LoRA-default.ipynb`](LoRA-default.ipynb) | **Notebook Fine-Tuning Utama (Versi Diperbaiki)** | File yang telah diperbaiki agar instalasi, training, penyimpanan, merge, dan inference dapat berjalan. File ini menjalankan satu konfigurasi LoRA standar. |
| [`LoRA-exp.ipynb`](LoRA-exp.ipynb) | **Notebook Eksperimen Hyperparameter LoRA** | File untuk menjalankan beberapa run dengan perubahan rank, alpha, dropout, dan target modules, lalu membandingkan training loss. |

Dengan demikian, `section4.ipynb` berfungsi sebagai **referensi awal**, `LoRA-default.ipynb` sebagai **pipeline utama yang sudah diperbaiki**, dan `LoRA-exp.ipynb` sebagai **pipeline eksperimen**. Hasil eksperimen dalam laporan ini terutama merujuk pada dua file terakhir.

## Pendahuluan

LoRA (Low-Rank Adaptation) adalah teknik fine-tuning yang membekukan bobot model dasar dan hanya melatih matriks tambahan berukuran kecil pada layer tertentu. Full fine-tuning memperbarui seluruh bobot model, sehingga membutuhkan memori, waktu, dan ruang penyimpanan jauh lebih besar. Pada LoRA, adapter dapat disimpan secara terpisah dan digabungkan kembali dengan model dasar ketika diperlukan. Efisiensi ini membuat LoRA cocok untuk perangkat dengan sumber daya terbatas. Kekurangannya, kemampuan adaptasi bergantung pada rank, alpha, dropout, layer target, kualitas dataset, dan durasi training. Pada tugas ini, LoRA digunakan untuk menyesuaikan `HuggingFaceTB/SmolLM2-135M` dengan dataset percakapan `HuggingFaceTB/smoltalk`.

## Perubahan dari `section4.ipynb`

`section4.ipynb` mengalami beberapa error ketika dijalankan di Google Colab. Perbaikan yang diterapkan pada `LoRA-default.ipynb` meliputi kompatibilitas `warmup_ratio`/`warmup_steps`, pemilihan `bf16` berdasarkan hardware, perubahan nama argumen `SFTTrainer`, penanganan `torchao`, fallback `chat_template`, serta perbaikan lokasi adapter dan model hasil merge. Notebook juga menambahkan progress training agar step, loss, epoch, learning rate, waktu, dan ETA dapat dipantau.

### Rincian perbedaan notebook

| Bagian | `section4.ipynb` | `LoRA-default.ipynb` | Dampak perbaikan |
|---|---|---|---|
| Instalasi library | Hanya memasang library utama. | Memasang library utama dan `torchao>=0.16.0`. | Menghindari konflik versi `peft` dan `torchao`. |
| Warmup scheduler | Langsung memakai `warmup_ratio`. | Memeriksa API `SFTConfig`, lalu memakai `warmup_ratio` atau menghitung `warmup_steps`. | Kompatibel dengan beberapa versi TRL. |
| Precision | `bf16=True` secara tetap. | `bf16` hanya aktif jika CUDA dan bfloat16 didukung. | Dapat berjalan pada CPU atau GPU tanpa dukungan bf16. |
| Argumen panjang sequence | Menggunakan `max_seq_length` langsung pada `SFTTrainer`. | Memeriksa dukungan `max_seq_length`, `max_length`, atau atribut `SFTConfig`. | Menghindari `TypeError` akibat perubahan API TRL. |
| Tokenizer trainer | Selalu memakai argumen `tokenizer`. | Memilih `tokenizer` atau `processing_class`. | Kompatibel dengan API TRL lama dan baru. |
| Packing dataset | Selalu mengirim `packing=True` dan `dataset_kwargs`. | Mengirimnya hanya jika didukung, atau mengatur `args.packing`. | Menghindari error argumen tidak dikenal. |
| Chat template | Mengasumsikan tokenizer sudah memiliki template. | Menambahkan fallback `chat_template` jika nilainya kosong. | Dataset percakapan dapat diproses oleh `SFTTrainer`. |
| Training monitoring | Hanya memanggil `trainer.train()`. | Menampilkan progress, step, loss, epoch, learning rate, waktu, dan ETA. | Proses training dapat dipantau secara langsung. |
| Penyimpanan dan merge | Menggunakan `args.output_dir` secara langsung. | Menyimpan adapter lalu menggabungkan model dari folder lokal yang benar. | Tidak lagi menganggap folder lokal sebagai repository Hugging Face. |
| Pembersihan memori | `del model` dan `del trainer` tanpa pemeriksaan. | Menghapus objek hanya jika tersedia dan menjalankan garbage collection. | Tidak muncul `NameError` saat objek sudah dihapus. |

Perbedaan tersebut menunjukkan bahwa perubahan bukan hanya perubahan hyperparameter. Sebagian besar perubahan diperlukan agar notebook yang awalnya dibuat untuk versi library tertentu dapat berjalan pada lingkungan Google Colab dengan versi library yang berbeda.

### Template screenshot laporan

Tambahkan screenshot pada lokasi berikut. Simpan gambar dalam folder `screenshots/` agar struktur pengumpulan tetap rapi.

#### Screenshot 1 — Error dari notebook awal

**Tujuan:** menunjukkan masalah saat `section4.ipynb` dijalankan.

```text
[TEMPEL SCREENSHOT ERROR SECTION4 DI SINI]
Caption: Gambar 1. Error pada section4.ipynb saat dijalankan di Google Colab.
```

File yang disarankan: `screenshots/01-error-section4.png`

#### Screenshot 2 — Konfigurasi yang sudah diperbaiki

**Tujuan:** menunjukkan fallback API, pemilihan precision, atau konfigurasi training pada `LoRA-default.ipynb`.

```text
[TEMPEL SCREENSHOT KONFIGURASI PERBAIKAN DI SINI]
Caption: Gambar 2. Konfigurasi kompatibel pada LoRA-default.ipynb.
```

File yang disarankan: `screenshots/02-konfigurasi-perbaikan.png`

#### Screenshot 3 — Progress training

**Tujuan:** menunjukkan training berjalan dan loss berubah dari waktu ke waktu.

```text
[TEMPEL SCREENSHOT PROGRESS TRAINING DI SINI]
Caption: Gambar 3. Progress training yang menampilkan step, loss, epoch, dan ETA.
```

File yang disarankan: `screenshots/03-progress-training.png`

#### Screenshot 4 — Tabel hasil eksperimen

**Tujuan:** menunjukkan perbandingan `rank`, `alpha`, `dropout`, target modules, dan `train_loss`.

```text
[TEMPEL SCREENSHOT TABEL TRAINING LOSS DI SINI]
Caption: Gambar 4. Perbandingan hasil training dari beberapa konfigurasi LoRA.
```

File yang disarankan: `screenshots/04-tabel-training-loss.png`

#### Screenshot 5 — Perbandingan output inference

**Tujuan:** menunjukkan output empat prompt setelah fine-tuning. Jika baseline sudah dijalankan ulang, tampilkan baseline dan hasil fine-tuning berdampingan.

```text
[TEMPEL SCREENSHOT OUTPUT BASELINE DAN FINE-TUNING DI SINI]
Caption: Gambar 5. Perbandingan output model sebelum dan sesudah fine-tuning.
```

File yang disarankan: `screenshots/05-perbandingan-inference.png`

## Eksperimen

Konfigurasi umum semua run adalah satu epoch, learning rate `2e-4`, batch size `2`, gradient accumulation `2`, dan sequence length `1512`. `LoRA-default.ipynb` memakai konfigurasi standar `r=6`, `alpha=8`, `dropout=0.05`, dan target `all-linear`. `LoRA-exp.ipynb` menjalankan empat konfigurasi:

| Run | Rank | Alpha | Dropout | Target modules | Training loss |
|---|---:|---:|---:|---|---:|
| Default | 6 | 8 | 0.05 | `all-linear` | 1.8389 |
| `r4_a8_d005_all` | 4 | 8 | 0.05 | `all-linear` | 1.8291 |
| `r8_a16_d005_all` | 8 | 16 | 0.05 | `all-linear` | **1.6855** |
| `r8_a16_d010_qv` | 8 | 16 | 0.10 | `q_proj`, `v_proj` | 2.1138 |
| `r16_a32_d010_qv` | 16 | 32 | 0.10 | `q_proj`, `v_proj` | 1.9363 |

Rank menentukan kapasitas adapter, alpha mengatur skala pembaruan, dropout memberi regularisasi, dan target modules menentukan bagian model yang dilatih. Konfigurasi `r8_a16_d005_all` menghasilkan training loss terendah. Namun, karena belum ada validation split, hasil ini hanya menunjukkan kecocokan terhadap data training.

## Hasil: perbandingan sebelum dan sesudah fine-tuning

`section4.ipynb` tidak menyimpan output inference sebelum fine-tuning; cell inference-nya juga belum memiliki output tersimpan. Karena itu, kolom “sebelum” di bawah ini dicatat sebagai **baseline yang perlu dijalankan ulang**, bukan hasil pengukuran yang sudah terekam. Kolom “sesudah” berasal dari output model LoRA eksperimen terbaik yang tercatat di `LoRA-exp.ipynb`.

| Prompt | Sebelum fine-tuning | Sesudah fine-tuning |
|---|---|---|
| Ibu kota Jerman | Output baseline belum direkam pada `section4.ipynb`. | Menjawab Berlin, tetapi kemudian mencampur pertanyaan lain dan menghasilkan fakta yang tidak konsisten. |
| Fungsi factorial Python | Output baseline belum direkam. | Menjelaskan factorial secara umum, tetapi tidak memberikan fungsi Python. |
| Pagar 25 × 15 kaki | Output baseline belum direkam. | Menjawab 25 kaki; jawaban benar seharusnya `2 × (25 + 15) = 80` kaki. |
| Perbedaan buah dan sayur | Output baseline belum direkam. | Memberikan jawaban terkait, tetapi definisinya kabur dan beberapa bagian berulang. |

Untuk melengkapi perbandingan numerik, model dasar seharusnya dijalankan dengan empat prompt yang sama sebelum adapter dipasang, lalu output dan pengaturan generation disimpan.

## Analisis

Training loss terbaik diperoleh pada rank 8, alpha 16, dropout 0.05, dan `all-linear`. Target `q_proj`/`v_proj` berjalan lebih cepat, tetapi loss-nya lebih tinggi; pengurangan layer target menghemat komputasi dengan konsekuensi kapasitas adaptasi yang lebih kecil. Walaupun loss menurun, kualitas jawaban belum stabil. Model masih mengulang teks, mencampur beberapa percakapan, gagal memberi kode factorial, dan salah menghitung keliling. Temuan ini menunjukkan bahwa training loss bukan pengganti evaluasi instruction-following. Masalah juga kemungkinan dipengaruhi dataset percakapan, chat template, model yang kecil, training satu epoch, serta pengaturan generation yang sebelumnya memberi warning karena `max_length` dan `max_new_tokens` dipakai bersamaan. Evaluasi selanjutnya perlu memakai validation loss, baseline yang benar-benar terekam, temperature terkontrol, dan prompt yang sama untuk semua model.

## Kesimpulan

Perbaikan dari `section4.ipynb` membuat pipeline LoRA lebih dapat dijalankan di Google Colab dan memungkinkan eksperimen beberapa konfigurasi. Run terbaik berdasarkan training loss adalah `r8_a16_d005_all` dengan loss `1.6855`. Akan tetapi, output sesudah fine-tuning belum menunjukkan peningkatan instruction-following yang konsisten. Perbandingan sebelum-sesudah belum sepenuhnya terukur karena output baseline tidak tersimpan di notebook awal. Oleh sebab itu, model belum sebaiknya dinilai hanya dari training loss; diperlukan baseline inference, validation set, dan evaluasi lebih banyak.

## AI Usage Disclosure

AI digunakan sebagai asisten untuk menganalisis error notebook, membantu memperbaiki kompatibilitas library, menyiapkan variasi eksperimen LoRA, menyusun tabel, dan merapikan laporan. Nilai training loss serta ringkasan output diambil dari output notebook. Interpretasi, pemeriksaan jawaban, dan kesimpulan akhir harus diverifikasi oleh penulis.
