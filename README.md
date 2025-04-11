# Tubes1ML_G07_SSB
Tugas Besar 1 Pembelajaran Mesin | Feedforward Neural Network

## Deskripsi Singkat
Diagonal magic cube adalah salah satu dari enam kelas magic cube yang merupakan sebuah kubus yang terdiri dari angka-angka mulai dari 1 hingga Pada tugas besar ini, akan diimplementasikan sebuah modul Feedforward Neural Network (FFNN). Modul ini harus dapat menerima konfigurasi jumlah neuron pada setiap layer, mulai dari input layer hingga output layer, sehingga struktur jaringan dapat dirancang sesuai karakteristik permasalahan yang ingin diselesaikan.

Selain itu, setiap layer dalam jaringan dapat menggunakan fungsi aktivasi yang berbeda. Fungsi aktivasi yang perlu didukung mencakup Linear, ReLU, Sigmoid, Hyperbolic Tangent (tanh), dan Softmax. Untuk proses pelatihan, model juga harus bisa menggunakan beberapa jenis fungsi loss sesuai kebutuhan, yaitu Mean Squared Error (MSE), Binary Cross-Entropy, dan Categorical Cross-Entropy.

FFNN ini juga perlu memiliki beberapa metode inisialisasi bobot dan bias, termasuk inisialisasi nol (zero initialization), distribusi acak uniform, dan distribusi acak normal. Setiap model yang diinisialisasi harus dapat menyimpan nilai bobot dan bias dari setiap neuron, serta menyimpan nilai gradiennya untuk keperluan proses pelatihan.

Agar model ini dapat dianalisis dengan baik, perlu disediakan fitur visualisasi yang menampilkan struktur jaringan lengkap dengan bobot dan gradien masing-masing neuron dalam bentuk grafis. Selain itu, model juga harus memiliki fungsi untuk menampilkan distribusi bobot dan distribusi gradien dari tiap layer, sehingga memudahkan dalam memahami dinamika pelatihan. Model juga harus memiliki kemampuan untuk disimpan (save) dan dimuat kembali (load) agar dapat digunakan ulang tanpa pelatihan dari awal.

Secara fungsional, model ini wajib mendukung proses forward propagation untuk data dalam bentuk batch, serta backward propagation untuk menghitung gradien berdasarkan data batch tersebut. Setelah gradien dihitung, bobot diperbarui menggunakan metode gradient descent. Untuk melatih model, pengguna harus bisa mengatur beberapa parameter penting seperti batch size, learning rate, jumlah epoch, dan opsi verbose untuk menampilkan progress pelatihan. Proses pelatihan ini akan menghasilkan histori berupa training loss dan validation loss pada setiap epoch, yang dapat digunakan untuk mengevaluasi performa model selama proses pelatihan.

## Cara Set Up dan Run
1. Clone repository berikut
2. Buat file ipynb pada folder test
3. Tambahkan ```sys.path.append(os.path.join(os.getcwd(), "../src"))``` sebelum melakukan ```from ffnn import FFNN```
4. Panggil kelas dan fungsi sesuai dengan kubutuhan

## Pembagian Tugas
1. 13522060: Visualisasi
2. 13522062: Forward
3. 13522082: Backward