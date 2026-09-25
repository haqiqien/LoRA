# Laporan Eksperimen Fine-Tuning LoRA

Notebook yang digunakan:

- [`LoRA-default.ipynb`](LoRA-default.ipynb): satu kali fine-tuning dengan konfigurasi LoRA default.
- [`LoRA-exp.ipynb`](LoRA-exp.ipynb): empat run dengan variasi parameter LoRA.

## Pendahuluan

LoRA (Low-Rank Adaptation) adalah metode fine-tuning yang membekukan bobot model dasar, lalu menambahkan matriks kecil yang dapat dilatih pada bagian tertentu dari model. Dengan demikian, proses training hanya memperbarui sejumlah kecil parameter tambahan. Pada full fine-tuning, seluruh bobot model ikut diperbarui sehingga kebutuhan memori, waktu, dan penyimpanan jauh lebih besar. LoRA menghasilkan adapter berukuran kecil yang dapat disimpan atau diganti tanpa menyalin seluruh model. Konsekuensinya, LoRA lebih hemat sumber daya, tetapi kemampuan adaptasinya bergantung pada rank, lokasi layer yang dipilih, alpha, dropout, dan jumlah data. Dalam eksperimen ini, LoRA digunakan untuk menyesuaikan `SmolLM2-135M` pada dataset percakapan `HuggingFaceTB/smoltalk`.

## Eksperimen

Konfigurasi umum yang digunakan pada kedua notebook adalah satu epoch, learning rate `2e-4`, batch size per device `2`, gradient accumulation `2`, dan sequence length `1512`. `LoRA-default.ipynb` menggunakan rank `6`, alpha `8`, dropout `0.05`, dan target `all-linear`.

### Metodologi dan fungsi parameter

Model dasar yang digunakan adalah `HuggingFaceTB/SmolLM2-135M`. Dataset `HuggingFaceTB/smoltalk` dengan konfigurasi `everyday-conversations` digunakan sebagai data training. Training dilakukan menggunakan `SFTTrainer` dan adapter PEFT LoRA selama satu epoch.

- **Rank (`r`)** menentukan ukuran matriks low-rank. Rank lebih besar memberi kapasitas adaptasi lebih tinggi, tetapi menambah parameter dan kebutuhan memori.
- **Alpha** menentukan skala pembaruan LoRA. Nilai lebih besar dapat memperkuat pengaruh adapter terhadap model dasar.
- **Dropout** membantu mengurangi overfitting pada adapter. Nilai `0.10` memberikan regularisasi lebih kuat daripada `0.05`.
- **Target modules** menentukan layer yang diberi adapter. `all-linear` mencakup layer linear yang sesuai, sedangkan `q_proj` dan `v_proj` hanya menargetkan proyeksi query dan value.

Training loss merupakan rata-rata loss pada data training selama satu epoch. Karena eksperimen tidak menggunakan validation split, nilai ini hanya digunakan untuk perbandingan awal dan belum membuktikan kemampuan generalisasi.

### Training loss

| Sumber | Run | Rank | Alpha | Dropout | Target modules | Training loss |
|---|---|---:|---:|---:|---|---:|
| Default | konfigurasi standar | 6 | 8 | 0.05 | `all-linear` | **1.8389** |
| Eksperimen | `r4_a8_d005_all` | 4 | 8 | 0.05 | `all-linear` | 1.8291 |
| Eksperimen | `r8_a16_d005_all` | 8 | 16 | 0.05 | `all-linear` | **1.6855** |
| Eksperimen | `r8_a16_d010_qv` | 8 | 16 | 0.10 | `q_proj`, `v_proj` | 2.1138 |
| Eksperimen | `r16_a32_d010_qv` | 16 | 32 | 0.10 | `q_proj`, `v_proj` | 1.9363 |

Setiap run eksperimen memuat ulang model dasar, membuat adapter baru, dan menyimpan hasil pada folder terpisah. Berdasarkan training loss, konfigurasi terbaik adalah `r8_a16_d005_all` dengan loss `1.6855`. Nilai ini hanya menunjukkan performa pada data training; belum dapat dianggap sebagai bukti generalisasi terbaik tanpa validation loss atau evaluasi pada dataset terpisah.

### Runtime eksperimen

| Run | Runtime | Kecepatan |
|---|---:|---:|
| `r4_a8_d005_all` | 512.28 detik | 0.146 step/detik |
| `r8_a16_d005_all` | 513.02 detik | 0.146 step/detik |
| `r8_a16_d010_qv` | 452.13 detik | 0.166 step/detik |
| `r16_a32_d010_qv` | 452.48 detik | 0.166 step/detik |

Target `q_proj` dan `v_proj` berjalan lebih cepat karena jumlah layer yang diberi adapter lebih sedikit. Runtime yang lebih singkat tidak berarti hasilnya lebih baik; pada eksperimen ini kedua konfigurasi tersebut memiliki training loss lebih tinggi.

### Dokumentasi proses eksperimen

> **Tempat Screenshot 1 — Progress training**  
> Ambil screenshot saat progress bar menampilkan step, epoch, loss, learning rate, elapsed time, dan ETA. Simpan sebagai `screenshots/progress-training.png`.

_Screenshot progress training dapat ditempatkan di sini._

> **Tempat Screenshot 2 — Tabel training loss**  
> Ambil screenshot output `results_table` yang memuat nama run, parameter LoRA, `train_loss`, dan runtime. Simpan sebagai `screenshots/training-loss-table.png`.

_Screenshot tabel training loss dapat ditempatkan di sini._

## Hasil

Output berikut diringkas dari empat prompt yang dijalankan setelah model hasil eksperimen terbaik digabungkan. Sebagai pembanding, perilaku model default juga dicatat dari output notebook.

Model eksperimen terbaik dipilih berdasarkan training loss terendah, yaitu `r8_a16_d005_all`. Pemilihan ini bersifat sementara karena belum menggunakan validation loss. Pengaturan generation juga memengaruhi hasil, sehingga output digunakan sebagai observasi kualitatif, bukan skor evaluasi formal.

| Prompt | `LoRA-default.ipynb` | Model eksperimen terbaik |
|---|---|---|
| Ibu kota Jerman | Tidak menjawab langsung dengan stabil; menghasilkan rangkaian jawaban berulang dan klaim sejarah yang tidak tepat. | Menjawab “Berlin”, tetapi dilanjutkan dengan teks dari beberapa percakapan lain dan fakta yang tidak konsisten. |
| Fungsi factorial Python | Tidak menghasilkan fungsi Python yang dapat digunakan; banyak pengulangan prompt. | Menjelaskan definisi factorial, tetapi tidak memberikan kode fungsi Python. |
| Pagar taman 25 × 15 kaki | Jawaban berulang dan salah; beberapa angka seperti 220 dan 2420 muncul. | Menjawab 25 kaki, masih salah. Jawaban matematis yang benar adalah `2 × (25 + 15) = 80` kaki. |
| Perbedaan buah dan sayur | Mengulang prompt dan token percakapan. | Memberikan penjelasan sederhana, tetapi definisinya masih kabur dan kembali menghasilkan pengulangan. |

### Interpretasi tiap prompt

1. **Ibu kota Jerman**: model eksperimen berhasil menyebut Berlin, tetapi jawaban kemudian tercampur dengan pertanyaan lain dan fakta yang tidak konsisten.
2. **Fungsi factorial**: model memahami topik secara umum, tetapi tidak memenuhi instruksi karena tidak memberikan kode Python.
3. **Soal pagar**: jawaban benar adalah `2 x (25 + 15) = 80` kaki. Kedua model gagal menghitung keliling dengan benar.
4. **Buah dan sayur**: model memberikan jawaban yang berhubungan dengan prompt, tetapi definisi masih kabur dan keluaran mengandung pengulangan.

### Dokumentasi output inference

> **Tempat Screenshot 3 — Output empat prompt**  
> Ambil screenshot cell inference yang menampilkan prompt dan response keempat pertanyaan. Pastikan jawaban yang berulang atau salah tetap terlihat sebagai bukti evaluasi. Simpan sebagai `screenshots/inference-results.png`.

_Screenshot output empat prompt dapat ditempatkan di sini._

## Analisis

1. `r8_a16_d005_all` memberikan training loss paling rendah. Ini menunjukkan kapasitas rank yang lebih besar dan alpha yang lebih tinggi membantu adaptasi pada konfigurasi `all-linear` dalam eksperimen ini.
2. Target `q_proj` dan `v_proj` dengan dropout `0.10` menghasilkan loss lebih tinggi daripada `all-linear`. Namun, target yang lebih terbatas dapat menghemat parameter; trade-off tersebut belum diuji dengan metrik validasi atau ukuran adapter.
3. Training loss yang rendah tidak otomatis menghasilkan jawaban yang benar. Model eksperimen masih salah pada perhitungan keliling dan tidak mampu menghasilkan kode factorial.
4. Banyak output berisi pengulangan, token internal, dan potongan percakapan lain. Hal ini mengindikasikan format chat template, preprocessing dataset, atau konfigurasi generation masih perlu diperbaiki.
5. Evaluasi saat ini bersifat kualitatif dan hanya menggunakan empat prompt. Eksperimen lanjutan sebaiknya menambahkan validation split, `eval_loss`, `max_new_tokens` yang eksplisit, temperature yang terkendali, serta pembersihan format output.
6. Warning generation menunjukkan `max_new_tokens` dan `max_length` digunakan bersamaan. Eksperimen berikutnya sebaiknya hanya menggunakan `max_new_tokens` agar panjang keluaran lebih mudah dikendalikan.
7. Model berukuran kecil dan hanya dilatih satu epoch, sehingga hasil berulang atau kurang akurat masih mungkin terjadi. Menambah data instruksi yang bersih dan memperbaiki format chat kemungkinan lebih berdampak daripada hanya menaikkan rank.

## Kesimpulan

LoRA berhasil menjalankan fine-tuning dengan biaya parameter dan penyimpanan yang lebih ringan dibandingkan full fine-tuning. Dari empat konfigurasi yang diuji, `r8_a16_d005_all` memperoleh training loss terendah, yaitu `1.6855`. Namun, hasil generasi belum konsisten: model masih mengulang teks, mencampur percakapan, dan melakukan kesalahan faktual maupun aritmetika. Karena itu, konfigurasi dengan training loss terendah belum dapat dinyatakan sebagai model terbaik secara umum. Diperlukan validation loss, evaluasi lebih banyak, dan perbaikan preprocessing serta generation sebelum model digunakan lebih lanjut.

## AI Usage Disclosure

AI digunakan sebagai asisten teknis untuk membantu membaca struktur notebook, memperbaiki kompatibilitas API `TRL`/`PEFT`, menambahkan eksperimen konfigurasi LoRA, memperbaiki alur penyimpanan dan inference, serta menyusun draf laporan ini. Nilai training loss dan ringkasan output berasal dari output notebook. Interpretasi hasil, pemeriksaan kesalahan jawaban, dan keputusan akhir mengenai kesimpulan perlu diverifikasi oleh penulis.
