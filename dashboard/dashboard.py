import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import f_oneway

# Load Data
data_url = "https://github.com/dwndnt/Analisis-Data-Bike-Sharing/blob/main/dashboard/hour.csv"
hour = pd.read_csv(data_url)

st.title("Bike Sharing Data Analysis")

# checkbox untuk show/hide data
show_data = st.checkbox("Tampilkan Data")

if show_data:
    st.subheader("Data Peminjaman Sepeda")
    st.write(hour)  # Menampilkan data jika checkbox dicentang


# Defenisikan kolom numerik
numeric_columns = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']

# fungsi untuk identifikasi outlier
def detect_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

st.header("Data Wrangling")
st.subheader("Identifikasi Outlier")
# membuat plot outlier
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Outlier Kolom Numerik')

for i, column in enumerate(numeric_columns):
    row = i // 4
    col = i % 4
    axes[row, col].boxplot(hour[column])
    axes[row, col].set_title(column)
    axes[row, col].set_xlabel(column)

plt.tight_layout()
st.pyplot(fig)

st.subheader("Proses Imputasi")
st.markdown("Berdasarkan hasil analisa boxplot, terlihat bahwa data numerik memiliki pecilan yang lumayan besar. Sehingga, perlu diatasi menggunakan metode Imputasi. formulasi imputasi yang digunakan :")

# Definisikan rumus LaTeX
latex_formula = r"""
\begin{align*}
Q1 &= \text{data[column].quantile}(0.25) \\
Q3 &= \text{data[column].quantile}(0.75) \\
IQR &= Q3 - Q1 \\
\text{lower\_bound} &= Q1 - 1.5 \times IQR \\
\text{upper\_bound} &= Q3 + 1.5 \times IQR \\
\end{align*}
"""

# Tampilkan rumus LaTeX menggunakan st.latex
st.latex(latex_formula)
# Mengatasi outlier pada kolom 'hum' dan 'windspeed' dengan metode imputasi
def impute_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Imputasi outlier dengan batas atas dan batas bawah
    data[column] = data[column].apply(lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x))

# Imputasi outlier pada kolom 'hum' dan 'windspeed'
impute_outliers(hour, 'hum')
impute_outliers(hour, 'windspeed')
impute_outliers(hour,'casual')
impute_outliers(hour,'registered')
impute_outliers(hour,'cnt')

# Membuat plot imputasi
fig, axes = plt.subplots(1, 5, figsize=(15, 6))

# Box plot 'hum'
axes[0].boxplot(hour['hum'])
axes[0].set_title('Box Plot - hum')

# Box plot 'windspeed'
axes[1].boxplot(hour['windspeed'])
axes[1].set_title('Box Plot - windspeed')

# Box plot 'casual'
axes[2].boxplot(hour['casual'])
axes[2].set_title('Box Plot - casual')

# Box plot 'registered'
axes[3].boxplot(hour['registered'])
axes[3].set_title('Box Plot - registered')

# Box plot 'cnt'
axes[4].boxplot(hour['cnt'])
axes[4].set_title('Box Plot - cnt')

st.pyplot(fig)

# Page for Exploratory Data Analysis
st.header("Exploratory Data Analysis")

# Hitung korelasi antara variabel cuaca dan jumlah peminjaman sepeda
correlation_matrix = hour[['temp', 'hum', 'windspeed', 'cnt']].corr()

# Tampilkan heatmap untuk visualisasi korelasi
st.subheader("Korelasi antara Unsur Cuaca dan Jumlah Peminjaman Sepeda")

st.markdown("Korelasi berkisar antara -1 hingga 1; di mana 1 menunjukkan korelasi positif sempurna, 0 menunjukkan tidak ada korelasi, dan -1 menunjukkan korelasi negatif sempurna.")

st.markdown("Korelasi antara **temp** (temperature) dan **cnt** (total count) adalah sekitar 0.404772. Ini menunjukkan adanya korelasi positif yang sedang antara suhu dan jumlah total peminjaman sepeda. Artinya, semakin tinggi suhu, semakin tinggi juga jumlah peminjaman sepeda.")

st.markdown("Korelasi antara **hum** (humidity) dan **cnt** adalah sekitar -0.322911. Ini menunjukkan adanya korelasi negatif yang sedang antara kelembapan udara dan jumlah total peminjaman sepeda. Artinya, semakin tinggi kelembapan, semakin rendah jumlah peminjaman sepeda.")

st.markdown("Korelasi antara **windspeed** dan **cnt** adalah sekitar 0.100906. Ini menunjukkan adanya korelasi positif yang lemah antara kecepatan angin dan jumlah total peminjaman sepeda. Meskipun positif, korelasi ini sangat lemah sehingga tidak ada hubungan yang kuat antara kecepatan angin dan peminjaman sepeda.")

st.write("Heatmap Korelasi:")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
plt.title('Korelasi antara Cuaca dan Jumlah Peminjaman Sepeda')
st.pyplot(fig)

# Pisahkan data antara hari libur dan hari kerja
data_libur = hour[hour['holiday'] == 1]
data_kerja = hour[hour['workingday'] == 1]

# Aggregasi data untuk peminjaman sepeda casual dan registered
peminjaman_libur = data_libur[['casual', 'registered', 'cnt']].mean()
peminjaman_kerja = data_kerja[['casual', 'registered', 'cnt']].mean()

st.header("Grafik Peminjaman Sepeda")
# Tampilkan peminjaman Casual, Registered, dan Total untuk hari libur
st.subheader("Peminjaman Sepeda pada Hari Libur")
st.bar_chart(peminjaman_libur)

# Tampilkan peminjaman Casual, Registered, dan Total untuk hari kerja
st.subheader("Peminjaman Sepeda pada Hari Kerja")
st.bar_chart(peminjaman_kerja)


st.title("UJI STATISTIK T-INDEPENDEN ")
# Ambil kolom peminjaman sepeda casual untuk uji t-independent
peminjaman_casual_libur = data_libur['casual']
peminjaman_casual_kerja = data_kerja['casual']

# Lakukan uji t independen
statistic, p_value = ttest_ind(peminjaman_casual_libur, peminjaman_casual_kerja)

st.subheader("Uji T-Independent untuk Peminjaman Sepeda Casual")
st.write("pengujian statistik t-independent untuk menguji apakah terdapat perbedaan yang signifikan dalam tipe peminjaman sepeda casual antara hari libur dan hari kerja.")
st.write("Hipotesis: Terdapat perbedaan yang signifikan antara tipe peminjaman sepeda casual pada hari libur dan hari kerja.")
st.write("Uji Statistik:")
st.write("Statistic:", statistic)
st.write("P-Value:", p_value)

alpha = 0.05  # Tingkat signifikansi

if p_value < alpha:
    st.write("Keputusan : Terdapat perbedaan yang signifikan antara tipe peminjaman sepeda casual pada hari libur dan hari kerja.")
else:
    st.write("Keputusan : Tidak terdapat perbedaan yang signifikan antara tipe peminjaman sepeda casual pada hari libur dan hari kerja.")
# Data
data = [hour['temp'], hour['hum'], hour['windspeed']]
labels = ['Temperature (temp)', 'Humidity (hum)', 'Windspeed (windspeed)']

# Boxplot

st.title("UJI STATISTIK ANOVA ")
# Definisi kelompok (variabel independen) dan variabel dependen
group_temp = hour['temp']
group_hum = hour['hum']
group_windspeed = hour['windspeed']
dependent_variable = hour['cnt']

# Melakukan uji ANOVA
statistic_temp, p_value_temp = f_oneway(dependent_variable, group_temp)
statistic_hum, p_value_hum = f_oneway(dependent_variable, group_hum)
statistic_windspeed, p_value_windspeed = f_oneway(dependent_variable, group_windspeed)

# Tingkat signifikansi (alpha)
alpha = 0.05

st.subheader("Hasil Uji Statistik ANOVA")

# Membuat data frame untuk menampilkan hasil uji statistik ANOVA dalam bentuk tabel
anova_results = pd.DataFrame({
    'Variabel Independen': ['temp', 'hum', 'windspeed'],
    'Statistic': [statistic_temp, statistic_hum, statistic_windspeed],
    'P-Value': [p_value_temp, p_value_hum, p_value_windspeed]
})

st.table(anova_results)

# Melakukan pengujian hipotesis
st.subheader("Hasil Pengujian Hipotesis")

if p_value_temp < alpha:
    st.write("Terdapat perbedaan yang signifikan antara temp dan cnt.")
else:
    st.write("Tidak terdapat perbedaan yang signifikan antara temp dan cnt.")

if p_value_hum < alpha:
    st.write("Terdapat perbedaan yang signifikan antara hum dan cnt.")
else:
    st.write("Tidak terdapat perbedaan yang signifikan antara hum dan cnt.")

if p_value_windspeed < alpha:
    st.write("Terdapat perbedaan yang signifikan antara windspeed dan cnt.")
else:
    st.write("Tidak terdapat perbedaan yang signifikan antara windspeed dan cnt.")

st.subheader("Visualisasi ANOVA")

fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot(data, labels=labels, vert=False)
ax.set_title('Hasil Uji ANOVA')
ax.set_xlabel('cnt (count/total)')
ax.set_ylabel('Variables')
ax.grid(axis='x')
st.pyplot(fig)


st.header(" Visualisasi & Explanatory Analysis")

# Pertanyaan 1: 

st.subheader("Pola Peminjaman Sepeda pada Hari Libur vs. Hari Kerja")
st.write("Berikut adalah perbandingan peminjaman sepeda pada hari libur dan hari kerja:")

data_holiday = hour[hour['holiday'] == 1]
data_working_day = hour[hour['workingday'] == 1]

rentals_holiday = data_holiday[['casual', 'registered', 'cnt']].mean()
rentals_working_day = data_working_day[['casual', 'registered', 'cnt']].mean()

fig, ax = plt.subplots()
categories = ['Casual', 'Registered', 'Total']
bar_width = 0.35
index = range(len(categories))

bar1 = ax.bar(index, rentals_holiday, bar_width, label='Holiday')
bar2 = ax.bar([i + bar_width for i in index], rentals_working_day, bar_width, label='Working Day')

ax.set_xlabel('Rental Category')
ax.set_ylabel('Average Count')
ax.set_title('Comparison of Bike Rentals on Holidays and Working Days')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(categories)
ax.legend()

st.pyplot(fig)

# Pertanyaan 2:

st.subheader("Kondisi temperatur, kelembaban, dan kecepatan angin terhadap pola peminjaman sepeda")
st.write("Berikut adalah visualisasi hubungan antara temperatur, kelembaban, dan kecepatan angin dengan jumlah peminjaman sepeda:")

# Scatter plots

st.subheader('kondisi temperatur terhadap pola peminjaman sepeda ')
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='temp', y='cnt', data=hour, alpha=0.5, ax=ax)
st.pyplot(fig)

st.subheader('kondisi kelembaban terhadap pola peminjaman sepeda ')
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='hum', y='cnt', data=hour, alpha=0.5, ax=ax)
st.pyplot(fig)


st.subheader('kondisi Kecepatan angin terhadap pola peminjaman sepeda ')
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='windspeed', y='cnt', data=hour, alpha=0.5, ax=ax)
st.pyplot(fig)


st.title("Conclusion")
st.write("Berdasarkan data rata-rata peminjaman sepeda pada hari libur dan hari kerja, dapat dilihat bahwa terdapat perbedaan yang signifikan dalam pola peminjaman sepeda antara kedua kondisi tersebut. Pada hari libur, rata-rata peminjaman sepeda casual mencapai 44.718, sedangkan peminjaman sepeda terdaftar (registered) mencapai 112.152. Dengan total peminjaman mencapai 156.87. Pada hari kerja, rata-rata peminjaman sepeda casual lebih rendah, yaitu sekitar 25.561, sementara peminjaman terdaftar lebih tinggi, mencapai 167.646. Total peminjaman pada hari kerja adalah 193.208. Pola ini menunjukkan bahwa pada hari libur, terdapat kontribusi yang lebih besar dari peminjaman casual, sementara pada hari kerja, peminjaman terdaftar mendominasi. Perbedaan ini mungkin disebabkan oleh perbedaan dalam rutinitas dan tujuan pengguna sepeda pada hari libur dan hari kerja. Pada hari libur, orang mungkin lebih cenderung untuk melakukan perjalanan rekreasi atau santai, yang dapat menjelaskan tingginya peminjaman casual. Sementara pada hari kerja, penggunaan sepeda mungkin lebih terkait dengan aktivitas sehari-hari seperti bersekolah atau pergi bekerja, yang menjelaskan dominasi peminjaman terdaftar. Dengan demikian, dapat disimpulkan bahwa terdapat perbedaan yang signifikan dalam pola peminjaman sepeda pada hari libur dan hari kerja, dengan peminjaman casual dan terdaftar berperan berbeda dalam masing-masing kondisi.")
st.write("Kondisi cuaca, termasuk temperatur (suhu), kelembaban, dan kecepatan angin, memainkan peran kunci dalam memengaruhi pola peminjaman sepeda dalam dataset ini. Analisis korelasi antara variabel cuaca dan jumlah total peminjaman sepeda (cnt) memberikan wawasan yang berharga tentang bagaimana faktor-faktor cuaca ini berdampak pada aktivitas bersepeda.")
st.write("Pertama, suhu atau temperatur memiliki pengaruh yang signifikan terhadap pola peminjaman sepeda. Korelasi positif yang sedang antara suhu dan cnt menunjukkan bahwa semakin tinggi suhu, semakin tinggi juga jumlah peminjaman sepeda. Ini mengindikasikan bahwa cuaca yang lebih hangat dan nyaman mendorong lebih banyak orang untuk bersepeda. Pada hari-hari yang cerah dan hangat, minat pengguna untuk mengambil sepeda sewaan tampaknya lebih tinggi.")
st.write("Di sisi lain, kelembaban udara memainkan peran yang berlawanan. Korelasi negatif yang sedang antara kelembaban udara dan cnt menunjukkan bahwa semakin tinggi kelembaban udara, semakin rendah jumlah peminjaman sepeda. Tingkat kelembapan yang tinggi dapat mengurangi minat orang untuk bersepeda, mungkin karena kondisi cuaca yang kurang nyaman dan pengaruh kelembapan terhadap kenyamanan bersepeda.")
st.write("Terakhir, kecepatan angin memiliki dampak yang lebih kecil, dengan korelasi positif yang lemah. Ini menunjukkan bahwa pengaruh kecepatan angin terhadap jumlah peminjaman sepeda tidak sekuat suhu atau kelembaban udara. Namun, dalam situasi tertentu, seperti cuaca yang sangat berangin atau buruk, kecepatan angin mungkin memiliki pengaruh lebih besar terhadap minat pengguna untuk bersepeda.")
st.write("Secara keseluruhan, faktor-faktor cuaca, terutama suhu dan kelembaban udara, memengaruhi pola peminjaman sepeda dengan cara yang signifikan. Ini memiliki implikasi penting dalam perencanaan dan manajemen sepeda sewaan, di mana pemahaman tentang hubungan ini dapat digunakan untuk mengoptimalkan layanan dan meningkatkan pengalaman pengguna terkait sepeda.")
