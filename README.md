# DataHeroes_JC_DS_AH_6_FinalProject

by  
- Bayu Andrianto Wirawan (bay.psil@gmail.com)
- Shafwan Hanif (shafwanh17@gmail.com) 

# DC Properties
*dataset ```DC Properties``` di download langsung dari website* https://www.kaggle.com/christophercorrea/dc-residential-properties

## WORKFLOW PROCESS

![Workflow](https://github.com/PurwadhikaDev/DataHeroes_JC_DS_AH_6_FinalProject/blob/main/img/FP-DCProp-Workflow.jpeg)


## 1. BUSINESS PROBLEMS

Kami adalah pebisnis Start Up bernama `AYH Inc` (Acquisition your House) yang bergerak di bidang jual beli rumah. `AYH Inc` memberikan alternatif solusi bagi pemilik rumah untuk menjual rumahnya dengan cepat dan mudah. Pemilik rumah dapat menjual rumahnya kepada kami untuk mendapatkan dana liquid yang cepat dibanding mengiklankannya di agency rumah ataupun online media.


Bisnis Properti merupakan salah satu bisnis yang dengan resiko yang cukup tinggi. Ketika kami membeli suatu properti untuk dijual kembali dalam jangka waktu yang cepat, kami haruslah mengetahui Harga Wajar dari Properti yang akan di beli untuk kemudian kami akan beli dibawah Harga Wajar tersebut berdasarkan resikonya untuk margin keuntungan kami. Selanjutnya ketika kami akan menjual harga property tersebut kami juga harus memperhatikan trend kenaikan harga property yang ada di pasar.


Dengan Machine Learning (ML), Pebisnis dapat dengan mudah mengetahi 3 hal di atas. sehingga Pebisnis dapat membeli suatu rumah dengan harga Wajar, dan dapat menjualnya dengan Profit maksimal.Dengan adanya ML ini, kami berharap dapat meminimalisir resiko dan memaksimalkan keuntungan dari setiap rumah yang akan dibeli dan kami jual kembali.

Sample Data yang yang dipakai adalah `DC Properties`, yaitu data rumah yang berada di Wasington DC. 

## 2. GOALS SETTINGS

Untuk dapat menentukan harga beli dan harga jual yang tepat kami setidaknya membutuhkan beberapa hal, yaitu:
1. Model regresi yang dapat memperkirakan harga wajar rumah di pasar
2. Model clustering yang dapat memberikan informasi cluster resiko rumah (Rumah resiko tinggi kami akan tawar 50% harga pasar, Rumah resiko sedang 70% dan rumah resiko rendah 80 % dari harga pasar)
3. Model time-series untuk memberikan informasi tren kenaikan harga 

Kami memformulasikan keekonomian bisnis kami sebagai berikut:
- Harga Wajar rumah yang ditawarkan kepada kami didapatkan dari hasil ML Regresi
- Harga beli yang kami akan bayar ke penjual merukapakan formula : ```Harga beli```  = ```Hasil regresi / Harga Wajar di Pasar``` x ```Penilaian Hasil cluster resiko```
- ```Harga Jual Rumah```  = ```Hasil regresi / Harga Wajar di Pasar``` x ```Tren Harga Rumah``` (Kami memiliki target setidaknya rumah harus laku 1 tahun sejak kami beli)
- Margin keuntungan kami adalah ```Harga Jual Rumah``` - ```Harga beli``` atau (```Hasil regresi / Harga Wajar di Pasar``` x ```Tren Harga Rumah```) - ```Harga beli```

Sehingga goals kami dalam project ini adalah:
- Mendapatkan analisis data DC Properties untuk mengetahui Prediksi Harga Properties di Wasington DC
- Mendapatkan analisis tentang pemetaan resiko dari harga jual di setiap wilayah
- Mengetahui trend harga rumah

## 3. DATA UNDERSTANDING

Dalam tahap EDA untuk Regressi ini, ada beberapa step dan output yang di harapakan, sehingga dapat memberika hasil maksimal dalam melakukan machine learning. 

Dalam data `DC_Properties` akan di lakukan EDA dengan 2 kelompok data `SOURCE`, karena terdapat 2 tipe Properties. Yaitu:

`RESIDENSIAL` vs `CONDOMINIUM`


![Data Perbandingan](https://github.com/PurwadhikaDev/DataHeroes_JC_DS_AH_6_FinalProject/blob/main/img/ResvsCon.png)

### Data Dictionary

* 	BATHRM	    : 	"Number of Full Bathrooms"
* 	HF_BATHRM	: 	Number of Half Bathrooms (no bathtub or shower)
* 	HEAT	    : 	Heating
* 	AC	        : 	Cooling
* 	NUM_UNITS	: 	Number of Units
* 	ROOMS	    : 	Number of Rooms
* 	BEDRM	: 	Number of Bedrooms
* 	AYB	: 	The earliest time the main portion of the building was built
* 	YR_RMDL	: 	Year structure was remodeled
* 	EYB	: 	The year an improvement was built more recent than actual year built
* 	STORIES	: 	Number of stories in primary dwelling
* 	SALEDATE	: 	Date of most recent sale
* 	PRICE	: 	Price of most recent sale
* 	QUALIFIED	: 	Qualified
* 	SALE_NUM	: 	Sale Number
* 	GBA	: 	Gross building area in square feet
* 	BLDG_NUM	: 	Building Number on Property
* 	STYLE	: 	Style
* 	STRUCT	: 	Structure
* 	GRADE	: 	Grade
* 	CNDTN	: 	Condition
* 	EXTWALL	: 	Extrerior wall
* 	ROOF	: 	Roof type
* 	INTWALL	: 	Interior wall
* 	KITCHENS	: 	Number of kitchens
* 	FIREPLACES	: 	Number of fireplaces
* 	USECODE	: 	Property use code
* 	LANDAREA	: 	Land area of property in square feet
* 	GIS_LAST_MOD_DTTM	: 	Last Modified Date
* 	SOURCE	: 	Raw Data Source
* 	CMPLX_NUM	: 	Complex number
* 	LIVING_GBA	: 	Gross building area in square feet
* 	FULLADDRESS	: 	Full Street Address
* 	CITY	: 	City
* 	STATE	: 	State
* 	ZIPCODE	: 	Zip Code
* 	NATIONALGRID	: 	Address location national grid coordinate spatial address
* 	LATITUDE	: 	Latitude
* 	LONGITUDE	: 	Longitude
* 	ASSESSMENT_NBHD	: 	Neighborhood ID
* 	ASSESSMENT_SUBNBHD	: 	Subneighborhood ID
* 	CENSUS_TRACT	: 	Census tract
* 	CENSUS_BLOCK	: 	Census block
* 	WARD	: 	"Ward (District is divided into eight wards, each with approximately 75,000 residents)
"
* 	SQUARE	: 	Square (from SSL - (Square, Suffix, Lot))
* 	X	: 	Latitude
* 	Y	: 	Longitude
* 	QUADRANT	: 	City quadrant (NE,SE,SW,NW)


## 4. DATA CLEANING, EDA
Beberapa step yang akan di lakukan adalah:

1. Load Dataset  
2. Menghitung Missing data
3. Data Prepocessing
    - Drop Data
    - Mengisi NaN dengan Data Lain
    - Mengisi NaN dengan Median
    - Mengisi NaN dengan Modus
4. Feature Enginering
5. Data Preprocessing
    - Dummy Variable
6. Normalized Price
    - melakukakan Normalisai Price - Kolom Price ditransform menggunakan Kubik
7. Output
8. Data Nominal
9. Data Visualization


Output Data yang dihasilkan Adalah
- Residensial
    1. Residensial for Reggresion `('Data/DC_Prop_Residential_Regression.csv')`
    2. Residensial for Clustering `('Data/DC_Prop_Residential_Clustering.csv')`
    3. Residensial for Time Series `('Data/DC_Prop_Residential_Time Series.csv')`
- Condominium
    1. Condominium for Reggresion `('Data/DC_Prop_Condominium_Regression.csv')`
    2. Condominium for Clustering `('Data/DC_Prop_Condominium_Clustering.csv')`
    3. Condominium for Time Series `('Data/DC_Prop_Condominium_Time Series.csv')`
    
    
  
  ## 5. MACHINE LEARNING MODELLING

Total ada `12 jenis model` yang kami eksplorasi untuk proyek ini (`8 Model Regresi` + `3 Model Clustering` + `1 Model Time Series`). 12 Model tersebut di-apply kepada `2 jenis property` yaitu `Property Rumah` dan `Property Condominium` sehingga `Total ada 24 Model` dalam proyek ini.

* `Regresi` untuk menghasilkan 2 model prediksi harga: Model Prediksi Harga Rumah dan Model Prediksi Harga Condominium
    - Mendapatkan analisis data DC Properties untuk mengetahui Prediksi Harga Properties di Wasington DC
        1.	OLS / Linear Regression
        2.	DecisionTreeRegressor
        3.	Random Forest
        4.	Adaboost
        5.	XGB Regressor
        6.	KNN
        7.	SVM
        8.	Ridge
        9.	Lasso

*  `Clustering` untuk menghasilkan 2 model clustering resiko: Model Resiko Rumah dan Model Resiko Condominium
    - menghitung resiko prosesntase harga jual (dipilih fitur yang kita perkirakan menjadi resiko menggunakan bisnis understanding)
    - Mendapatkan analisis tentang pemetaan resiko dari harga jual di setiap wilayah
        1.	KMeans
        2.	Agglomerative
        3.	Gaussian mixture model (GMM)


* `Time series` untuk menghasilkan 2 model tren harga: Tren kenaikan harga rumah dan Tren kenaikan harga condominium
    - Mengetahui trend harga rumah
        1.	Dekomposisi Tren

  
