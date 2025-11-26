============= CARA PENGGUNAAN ===============
1. sudah ada format di data_tembak.csv ini nanti yang dipakai model buat di train, nanti hasilnya offset_predictor.pkl
2. langsung di coba GUI.py buat makai model nya buat prediksi, nanti bakal muncul angka yang nanti dipake buat prediksi peluru jatuhnya
3. kalau mau implementasi di gamenya, ini pakai dynamic crosshair "nomogram classic top web" by stiv32

format file data_tembak.csv :
shell_travel_time, distance, angle, enemy_max_speed, enemy_actual_speed, offset_x

distance = Jarak ke musuh
angle = sudut kebentuk antara pandng aim dan arah laju musuh
shell_travel_time = Waktu tempuh peluru ke point aim yang bidik
enemy_max_speed= kecepatan laju musuh maximum
enemy_actual_speed = kecepatan saat pendataan
offset_x= titik di binocular dimana peluru jatuh (dinyatakan per jumlah garis di binocular)

untuk enemy_max_speed dan enemy_actual_speed, variabel ini memiliki perbedaan satuan yang belum bisa saya jelaskan, karena enemy_max_speed merupakan info asli dalam game ber metrik "knot", dan enemy_actual_speed merupakan hasil dari rumus :

enemy_actual_speed = (distance * offset_x) / (shell_travel_time * math.sin(math.radians(angle)))

di file csv itu ada 2 variabel : enemy_max_speed dan enemy_actual_speed ini dipake kalo pas mau buat yang lebih akurat aja, karena ini pake Random Forrest Regressor jadi dak perlu, kecuali kalau mau yang lebih scientific pakai rumus

model hanya memprediksi nilai offset_x (garis horizontal pada binocular), model ini tidak bisa memprediksi nilai offset_y (garis vertical pada binocular)

model ini hanya memprediksi, bukan menghitung secara akurat, jika ingin menghitung secara akurat, pakai rumus yang lebih scientific.