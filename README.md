# Fine-Tuning LLM dengan LoRA

Repositori ini berisi notebook untuk melakukan fine-tuning model bahasa menggunakan **LoRA (Low-Rank Adaptation)** dan library Hugging Face, termasuk `transformers`, `datasets`, `trl`, dan `peft`.

## Buka di Google Colab

[![Buka di Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haqiqien/LoRA/blob/main/LoRA-default.ipynb)

Atau buka langsung:

<https://colab.research.google.com/github/haqiqien/LoRA/blob/main/LoRA-default.ipynb>

## Isi notebook

Notebook [`LoRA-default.ipynb`](LoRA-default.ipynb) mencakup:

- Persiapan environment dan library yang diperlukan.
- Pemanggilan dataset dari Hugging Face.
- Fine-tuning `HuggingFaceTB/SmolLM2-135M` dengan LoRA dan `SFTTrainer`.
- Penyimpanan serta penggabungan adapter LoRA dengan model dasar.
- Pengujian model melalui inference sederhana.

## Cara menjalankan

1. Buka notebook melalui tautan Google Colab di atas.
2. Pilih runtime GPU melalui **Runtime - Change runtime type**.
3. Jalankan cell secara berurutan.
4. Saat diminta, lakukan login ke akun Hugging Face menggunakan access token.

> Pastikan akun Hugging Face memiliki izin untuk mengakses model atau dataset yang digunakan, serta jangan membagikan access token di dalam notebook.
