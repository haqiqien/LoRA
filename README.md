# Fine-Tuning LLM dengan LoRA

Repositori ini berisi dua notebook untuk melakukan fine-tuning model bahasa menggunakan **LoRA (Low-Rank Adaptation)** dengan library Hugging Face: `transformers`, `datasets`, `trl`, dan `peft`.

Model yang digunakan adalah `HuggingFaceTB/SmolLM2-135M`, sedangkan dataset contoh yang digunakan adalah `HuggingFaceTB/smoltalk` dengan konfigurasi `everyday-conversations`.

## Notebook

### 1. Fine-tuning standar

[`LoRA-default.ipynb`](LoRA-default.ipynb) menjalankan satu proses fine-tuning dengan satu konfigurasi LoRA.

[Buka LoRA-default.ipynb di Google Colab](https://colab.research.google.com/github/haqiqien/LoRA/blob/main/LoRA-default.ipynb)

Notebook ini mencakup:

- Instalasi library dan autentikasi Hugging Face.
- Pemanggilan dataset percakapan.
- Pemuatan model dan tokenizer.
- Konfigurasi LoRA dengan `SFTTrainer`.
- Penyesuaian API TRL untuk beberapa versi library.
- Pemilihan otomatis precision `bf16` hanya pada hardware yang mendukungnya.
- Progress training secara detail, termasuk step, loss, learning rate, epoch, waktu berjalan, dan ETA.
- Penyimpanan adapter LoRA dan pengujian inference.

### 2. Eksperimen beberapa konfigurasi

[`LoRA-exp.ipynb`](LoRA-exp.ipynb) digunakan untuk membandingkan beberapa konfigurasi LoRA dalam beberapa run terpisah.

[Buka LoRA-exp.ipynb di Google Colab](https://colab.research.google.com/github/haqiqien/LoRA/blob/main/LoRA-exp.ipynb)

Konfigurasi eksperimen yang tersedia:

| Run | Rank | Alpha | Dropout | Target modules |
|---|---:|---:|---:|---|
| `r4_a8_d005_all` | 4 | 8 | 0.05 | `all-linear` |
| `r8_a16_d005_all` | 8 | 16 | 0.05 | `all-linear` |
| `r8_a16_d010_qv` | 8 | 16 | 0.10 | `q_proj`, `v_proj` |
| `r16_a32_d010_qv` | 16 | 32 | 0.10 | `q_proj`, `v_proj` |

Setiap run memuat model baru, melatih adapter secara terpisah, menyimpan hasil ke folder berbeda, dan mencatat metrik training. Ringkasan hasil disimpan sebagai `lora_experiment_results.csv`.

## Perbedaan kedua notebook

| Aspek | `LoRA-default.ipynb` | `LoRA-exp.ipynb` |
|---|---|---|
| Tujuan | Fine-tuning satu model dengan satu konfigurasi | Membandingkan beberapa konfigurasi LoRA |
| Jumlah run | Satu run | Empat run berurutan |
| Parameter LoRA | Menggunakan konfigurasi default: rank 6, alpha 8, dropout 0.05 | Mengubah rank, alpha, dropout, dan target modules pada setiap run |
| Output | Satu folder model hasil fine-tuning | Satu folder untuk setiap konfigurasi eksperimen |
| Evaluasi | Menguji model hasil training melalui inference | Membandingkan loss dan waktu training dalam tabel hasil |
| Waktu dan memori | Lebih cepat dan ringan | Lebih lama dan membutuhkan memori lebih besar |

Gunakan `LoRA-default.ipynb` untuk mencoba alur fine-tuning dengan cepat. Gunakan `LoRA-exp.ipynb` jika ingin menganalisis pengaruh hyperparameter LoRA terhadap hasil training.

## Cara menjalankan

1. Buka salah satu notebook melalui Google Colab atau Jupyter.
2. Gunakan runtime GPU jika tersedia. Training CPU dapat berjalan lebih lambat.
3. Jalankan cell secara berurutan dari atas ke bawah.
4. Saat diminta, login ke Hugging Face menggunakan access token.
5. Jika library baru saja di-install, restart kernel/runtime sebelum menjalankan cell berikutnya.

Untuk eksperimen, jalankan seluruh cell hingga cell eksperimen dan tunggu semua run selesai. Empat run membutuhkan waktu dan memori lebih besar daripada `LoRA-default.ipynb`.

## Output

- Adapter/model hasil training disimpan di folder output yang ditentukan oleh `output_dir`.
- Notebook eksperimen membuat folder dengan pola `SmolLM2-FT-MyDataset-exp-<nama-run>`.
- Hasil perbandingan eksperimen disimpan di `lora_experiment_results.csv`.

## Catatan dependensi

Notebook memasang `torchao>=0.16.0` karena versi `peft` yang digunakan memerlukan versi tersebut. Setelah instalasi atau upgrade package, restart runtime agar versi package yang baru digunakan.

Pastikan akun Hugging Face memiliki izin untuk mengakses model atau dataset yang digunakan. Jangan membagikan access token di dalam notebook.
