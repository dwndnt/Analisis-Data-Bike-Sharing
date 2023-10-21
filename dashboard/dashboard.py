import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import f_oneway

# Load Data
hour = pd.read_csv("hour.csv")

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

st.subheader("Kondisi Unsur Cuaca terhadap pola peminjaman sepeda")

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