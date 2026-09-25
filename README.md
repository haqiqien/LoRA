# Eksperimen Fine-Tuning LLM dengan LoRA

Repository ini berisi pekerjaan fine-tuning `HuggingFaceTB/SmolLM2-135M` menggunakan LoRA, Hugging Face Transformers, Datasets, TRL, dan PEFT.

Repository GitHub: <https://github.com/haqiqien/LoRA>

Dataset yang digunakan adalah `HuggingFaceTB/smoltalk` dengan konfigurasi `everyday-conversations`.

## Struktur dan peran file

### `section4.ipynb` — Notebook Original Google Classroom

File awal yang diberikan melalui Google Classroom. Notebook ini menjadi referensi awal, tetapi mengalami beberapa error ketika dijalankan di Google Colab karena perbedaan versi library dan keterbatasan hardware runtime.

### `LoRA-default.ipynb` — Notebook Fine-Tuning Utama

Versi yang telah diperbaiki agar pipeline utama dapat berjalan. Notebook ini menjalankan satu fine-tuning dengan konfigurasi LoRA standar.

Perbaikan yang diterapkan meliputi:

- Kompatibilitas `warmup_ratio` dan `warmup_steps`.
- BF16 hanya diaktifkan jika hardware mendukung.
- Kompatibilitas argumen `SFTTrainer`, termasuk `max_length`, `max_seq_length`, `tokenizer`, dan `processing_class`.
- Penanganan `packing` dan `dataset_kwargs` berdasarkan versi TRL.
- Instalasi `torchao>=0.16.0` untuk kompatibilitas dengan PEFT.
- Fallback `chat_template` untuk tokenizer yang tidak memiliki template.
- Perbaikan path adapter, model merge, dan inference lokal.
- Progress training detail: step, loss, epoch, learning rate, elapsed time, dan ETA.

[Buka LoRA-default.ipynb di Google Colab](https://colab.research.google.com/github/haqiqien/LoRA/blob/main/LoRA-default.ipynb)

### `LoRA-exp.ipynb` — Notebook Eksperimen Hyperparameter

File ini digunakan untuk menjalankan beberapa eksperimen LoRA. Setiap run memuat model baru, mengubah parameter, menyimpan adapter di folder berbeda, dan mencatat training loss.

Konfigurasi yang dicoba:

| Run | Rank | Alpha | Dropout | Target modules | Training loss |
|---|---:|---:|---:|---|---:|
| Default | 6 | 8 | 0.05 | `all-linear` | 1.8389 |
| `r4_a8_d005_all` | 4 | 8 | 0.05 | `all-linear` | 1.8291 |
| `r8_a16_d005_all` | 8 | 16 | 0.05 | `all-linear` | **1.6855** |
| `r8_a16_d010_qv` | 8 | 16 | 0.10 | `q_proj`, `v_proj` | 2.1138 |
| `r16_a32_d010_qv` | 16 | 32 | 0.10 | `q_proj`, `v_proj` | 1.9363 |

Konfigurasi dengan training loss terendah adalah `r8_a16_d005_all`. Nilai ini hanya berasal dari data training karena eksperimen belum menggunakan validation split.

[Buka LoRA-exp.ipynb di Google Colab](https://colab.research.google.com/github/haqiqien/LoRA/blob/main/LoRA-exp.ipynb)

## Konfigurasi umum training

- Model: `HuggingFaceTB/SmolLM2-135M`
- Dataset: `HuggingFaceTB/smoltalk`
- Epoch: `1`
- Learning rate: `2e-4`
- Batch size per device: `2`
- Gradient accumulation: `2`
- Sequence length: `1512`
- Optimizer: AdamW
- Metode: supervised fine-tuning dengan `SFTTrainer` dan PEFT LoRA

## Cara menjalankan

1. Buka notebook melalui Google Colab.
2. Pilih runtime GPU jika tersedia.
3. Jalankan cell secara berurutan dari atas ke bawah.
4. Login ke Hugging Face jika diminta.
5. Setelah instalasi package selesai, restart runtime/kernel.
6. Jalankan ulang notebook dari awal.

Gunakan `LoRA-default.ipynb` untuk satu proses fine-tuning. Gunakan `LoRA-exp.ipynb` untuk menjalankan empat eksperimen secara berurutan. Notebook eksperimen membutuhkan waktu dan memori lebih besar.

## Output

Output utama yang dihasilkan:

- Folder adapter default sesuai `output_dir`.
- Folder eksperimen dengan pola `SmolLM2-FT-MyDataset-exp-<nama-run>`.
- File `lora_experiment_results.csv` berisi metrik setiap run.
- Folder `merged` untuk model LoRA yang sudah digabungkan dengan model dasar.
- Output inference dari empat prompt evaluasi.

## Laporan dan screenshot

- [Laporan Markdown](Laporan-Tugas-LoRA.md)
- [Laporan LaTeX](Laporan-Tugas-LoRA.tex)
- [Template screenshot](Template-Screenshot-LoRA.md)
- Folder [screenshots](screenshots/)

Laporan membahas perbedaan notebook original dan notebook hasil perbaikan, eksperimen parameter LoRA, training loss, hasil inference, analisis, kesimpulan, dan AI Usage Disclosure.

## Membuat PDF laporan

Pastikan MiKTeX atau TeX Live sudah terpasang. Dari Git Bash, jalankan:

```bash
cd "/c/Users/kyous/Downloads/En/s2/smt2/kapita selekta/LoRA"
mkdir -p pdf
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=pdf \
  Laporan-Tugas-LoRA.tex
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=pdf \
  Laporan-Tugas-LoRA.tex
```

PDF akan tersimpan di `pdf/Laporan-Tugas-LoRA.pdf`.

## Catatan keamanan

Jangan memasukkan Hugging Face access token ke dalam file notebook atau repository. Gunakan login Colab secara aman dan pastikan model/dataset dapat diakses oleh akun yang digunakan.
