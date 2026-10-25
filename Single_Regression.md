[laporan_praktikum_machine_learning_esp32.md](https://github.com/user-attachments/files/33929250/laporan_praktikum_machine_learning_esp32.md)
# Mission-06A — Linear Regression on ESP32

## Predicting Machine Temperature with MicroPython

**Mata kuliah:** Machine Learning  
**Topik:** Regresi linear sederhana, training, inference, dan evaluasi pada embedded system  
**Perangkat:** ESP32 (firmware MicroPython), kabel USB, laptop dengan Thonny IDE  
**Perangkat tambahan:** Tidak diperlukan sensor, breadboard, atau koneksi internet  
**Estimasi durasi:** 80–100 menit  
**Prasyarat:** Dasar Python, regresi linear, MAE, dan penggunaan Thonny untuk menjalankan MicroPython pada ESP32

---

## 1. Tujuan pembelajaran

Setelah praktikum, mahasiswa mampu:

1. Menjelaskan perbedaan **training** dan **inference**.
2. Mengimplementasikan model regresi linear sederhana pada ESP32 tanpa `numpy`, `pandas`, atau `scikit-learn`.
3. Menghitung koefisien slope dan intercept dari data training langsung pada ESP32.
4. Menguji model pada data yang **tidak digunakan saat training** dan menghitung *Mean Absolute Error* (MAE).
5. Mengidentifikasi risiko ekstrapolasi serta keterbatasan model regresi linear.

---

## 2. Studi kasus

Sebuah mesin hipotetis mengalami kenaikan temperatur seiring bertambahnya waktu operasi. Kita ingin memperkirakan temperatur berdasarkan waktu operasi.

| Variabel | Simbol | Satuan | Peran |
|---|---|---|---|
| Waktu operasi | `x` | menit | Fitur/input |
| Temperatur mesin | `y` | °C | Target/output |

Model regresi linear sederhana:

$$
\hat{y} = b_0 + b_1 x
$$

Keterangan:
- $b_0$: **intercept**, temperatur prediksi saat $x = 0$.
- $b_1$: **slope**, perubahan temperatur prediksi untuk tambahan satu menit operasi.
- $x$: waktu operasi mesin (menit).
- $\hat{y}$: temperatur hasil prediksi model (°C).

### Dataset sintetik

Berikut data **training** yang disimpan langsung di dalam kode ESP32:

| Waktu (menit) | Temperatur (°C) |
|---:|---:|
| 0 | 28.0 |
| 1 | 29.7 |
| 2 | 31.5 |
| 3 | 32.8 |
| 4 | 34.9 |
| 5 | 36.1 |
| 6 | 38.2 |
| 7 | 39.6 |
| 8 | 41.4 |

Data **testing** (tidak digunakan untuk menghitung koefisien):

| Waktu (menit) | Temperatur (°C) |
|---:|---:|
| 1.5 | 30.6 |
| 3.5 | 33.9 |
| 5.5 | 37.4 |
| 7.5 | 40.5 |

---

## 3. Persiapan

1. Sambungkan ESP32 ke laptop melalui USB.
2. Jalankan **Thonny** dan pilih interpreter **MicroPython (ESP32)** serta port serial yang benar.
3. Pastikan Shell Thonny dapat menjalankan `print("Hello ESP32")`.
4. Buat berkas baru untuk setiap eksperimen. Jalankan dari Thonny atau simpan berkas ke ESP32.
5. Untuk menghentikan `while True`, tekan **Stop** di Thonny atau `Ctrl+C` pada Shell.

---

## 4. Eksperimen A — Inference dengan koefisien yang sudah diketahui

Model linear yang ditentukan awal:
$$
\hat{T} = 28.0 + 1.7 t
$$

```python
# Mission-06A, Eksperimen A
b0 = 28.0       
b1 = 1.7        

def predict(minutes):
    return b0 + b1 * minutes

for minutes in [0, 2, 4, 6, 8]:
    predicted_temp = predict(minutes)
    print("Time:", minutes, "min | Prediction:",
          round(predicted_temp, 2), "C")
```

---

## 5. Eksperimen B — Training regresi linear langsung di ESP32

```python
# Mission-06A, Eksperimen B
train_data = [
    (0, 28.0), (1, 29.7), (2, 31.5),
    (3, 32.8), (4, 34.9), (5, 36.1),
    (6, 38.2), (7, 39.6), (8, 41.4)
]

def fit_linear_regression(data):
    n = len(data)
    if n < 2:
        raise ValueError("Need at least two training samples")

    mean_x = sum(row[0] for row in data) / n
    mean_y = sum(row[1] for row in data) / n

    numerator = sum((x - mean_x) * (y - mean_y) for x, y in data)
    denominator = sum((x - mean_x) ** 2 for x, y in data)

    if denominator == 0:
        raise ValueError("All training x values are identical")

    b1 = numerator / denominator
    b0 = mean_y - b1 * mean_x
    return b0, b1

def predict(x, b0, b1):
    return b0 + b1 * x

b0, b1 = fit_linear_regression(train_data)
print("Intercept:", round(b0, 4))
print("Slope:", round(b1, 4))
```

---

## 6. Eksperimen C — Mengukur error pada data testing

**Hasil Eksekusi Kode Evaluasi:**

| Koefisien Model | Nilai |
|---|---|
| Intercept ($b_0$) | 28.0222 |
| Slope ($b_1$) | 1.6667 |

**Tabel Perbandingan Prediksi dan Actual pada Data Test:**

| Waktu ($t$ menit) | Actual ($y$) | Predicted ($\hat{y}$) | Abs_error ($|y - \hat{y}|$) |
|---:|---:|---:|---:|
| 1.5 | 30.6 | 30.52 | 0.08 |
| 3.5 | 33.9 | 33.86 | 0.04 |
| 5.5 | 37.4 | 37.19 | 0.21 |
| 7.5 | 40.5 | 40.52 | 0.02 |

**NILAI MAE TEST:** `0.0889 °C`

**PENJELASAN MAKNA SATUAN (°C) PADA MAE:**

Penggunaan satuan derajat Celsius (°C) pada nilai MAE mengindikasikan besaran penyimpangan fisik rata-rata antara hasil perkiraan sistem dengan kondisi aktualnya dalam dimensi yang identik. Karena nilai MAE yang diperoleh sebesar $0.0889 ^\circ\text{C}$, hal ini menunjukkan bahwa secara statistik tingkat deviasi atau rata-rata galat prediksi yang dihasilkan oleh persamaan regresi linear mikrokontroler ini hanyalah berkisar $0.0889 ^\circ\text{C}$ dari suhu riil data uji.

---

## 7. Eksperimen D — Prediksi interaktif dari terminal serial

```python
while True:
    user_input = input("Waktu operasi (menit), atau q: ")
    if user_input.strip().lower() == "q":
        print("Program selesai")
        break
    try:
        minutes = float(user_input)
        if minutes < 0:
            print("Waktu tidak boleh negatif")
            continue
        predicted = predict(minutes, b0, b1)
        print("Predicted temperature:", round(predicted, 2), "C")
        if minutes > 8:
            print("WARNING: extrapolation beyond training range")
    except ValueError:
        print("Masukkan angka atau q")
```

---

## 8. Eksperimen E — Tantangan modifikasi dataset

### Tabel Hasil Pengujian 3 Skenario

| Skenario | Jumlah data training | Intercept (`b0`) | Slope (`b1`) | MAE testing (°C) |
|---|---:|---:|---:|---:|
| 1 — Asli | 9 | 28.0222 | 1.6667 | 0.0889 C |
| 2 — Data dikurangi | 6 | 27.9385 | 1.6654 | 0.1673 C |
| 3 — Satu outlier | 9 | 29.1444 | 1.6667 | 1.0444 C |

**ANALISIS:**

1. **Apakah volume dataset yang lebih melimpah secara mutlak menjamin peningkatan akurasi/kinerja model?**  
   *Jawab:* Tidak selalu. Meskipun Skenario 1 dan Skenario 3 sama-sama mengolah 9 sampel data, Skenario 3 justru menghasilkan nilai eror terburuk (MAE $1.0444 ^\circ\text{C}$). Hal ini menunjukkan bahwa **kualitas dan kebersihan data (data integrity)** jauh lebih krusial dibandingkan sekadar kuantitas data. Kumpulan data yang sedikit namun bersih (Skenario 2) memberikan akurasi yang lebih unggul dibandingkan data banyak tetapi terkontaminasi oleh pencilan (*outlier*).

2. **Mengapa kehadiran satu data outlier mampu merusak formulasi perhitungan OLS secara signifikan?**  
   *Jawab:* Prinsip dasar *Ordinary Least Squares* (OLS) berpatokan pada minimasi jumlah kuadrat selisih deviasi (squared residual). Penguadratan ini memberikan sanksi (bobot penalti) yang sangat besar pada titik data yang letaknya sangat jauh dari tren umum. Akibatnya, demi meminimalkan eror ekstrem dari satu titik *outlier* pada $t = 4$, garis regresi OLS dipaksa "mengalah" dan terdistorsi mengarah ke *outlier* tersebut. Hal ini menggeser nilai *intercept* ($b_0$) naik menjadi $29.1444$ dan menaikkan tingkat eror prediksi data normal secara drastis.

---

## 9. Ringkasan: Apa yang benar-benar dijalankan di ESP32?

| Bagian | Eksperimen A | Eksperimen B–E |
|---|---|---|
| Dataset training di ESP32 | Tidak diperlukan | Ya |
| Menghitung `b0` dan `b1` di ESP32 | Tidak | Ya |
| Inference/prediksi di ESP32 | Ya | Ya |
| Evaluasi pada test set | Tidak | Ya (C dan E) |
| Sensor fisik | Tidak | Tidak |
| Library machine learning | Tidak | Tidak |

---

## 10. Laporan singkat mahasiswa

### Jawaban Pertanyaan Refleksi

1. **Perbedaan mendasar antara proses Training dan Inference dalam konteks praktikum di ESP32:**
   - **Training:** Proses kalkulasi matematis berbasis algoritma OLS di mana mikrokontroler membaca sekumpulan *dataset* sampel lalu menghitung parameter *slope* ($b_1$) dan *intercept* ($b_0$). Tahap ini berfokus pada pembentukan pola (pembuatan model).
   - **Inference:** Proses kalkulasi operasional di mana ESP32 memanfaatkan nilai $b_0$ dan $b_1$ yang telah didapatkan sebelumnya untuk mengeksekusi rumus linear sederhana $\hat{y} = b_0 + b_1 x$ guna meramal nilai temperatur dari input waktu baru.

2. **Dapatkah nilai prediksi dihitung di luar rentang training, dan mengapa output tersebut berisiko tinggi (perlu diwaspadai)?**
   - **Dapat dihitung:** Secara matematis program akan tetap menghasilkan angka karena fungsi persamaan linear akan mengalikan nilai $x$ berapapun dengan koefisien yang ada.
   - **Risiko Ekstrapolasi:** Prediksi di luar batas rentang sampel ($x > 8$ menit) bertumpu pada asumsi hipotetis bahwa tren temperatur akan terus naik secara linear konstan tanpa batas. Padahal pada kondisi fisik nyata, komponen dapat mencapai titik jenuh thermal (suhu melandai) atau bahkan mengalami kerusakan. Oleh karena itu, output ekstrapolasi berisiko tinggi mengalami ketidakakuratan fatal.

3. **Analisis keuntungan dan keterbatasan kalkulasi training internal pada ESP32 dibandingkan sistem inference-only:**
   - **Keuntungan:** Sistem memiliki sifat adaptif lokal (*on-device learning*). Perangkat mampu menyesuaikan ulang parameter model secara mandiri di lapangan (edge) berdasarkan data spesifik tempat ia ditempatkan tanpa ketergantungan koneksi ke komputer luar atau cloud server.
   - **Keterbatasan:** Spesifikasi hardware ESP32 (komputasi CPU dan memori RAM) yang terbatas membatasi kompleksitas model. ESP32 tidak efisien jika dipaksa mengolah dataset berukuran sangat besar atau menjalankan algoritma ML kompleks yang membutuhkan kalkulasi matriks berat.

---

## 11. Kriteria keberhasilan

- [x] ESP32 menjalankan prediksi menggunakan koefisien tetap.
- [x] ESP32 berhasil menghitung slope dan intercept dari data training.
- [x] ESP32 menghasilkan MAE menggunakan data testing yang terpisah.
- [x] Mahasiswa mencoba input interaktif dan mengidentifikasi ekstrapolasi.
- [x] Mahasiswa membandingkan hasil tiga skenario dataset.
