# DataHeroes_JC_DS_AH_6_FinalProject

by  
- Bayu Andrianto Wirawan (bay.psil@gmail.com)
- Shafwan Hanif (shafwanh17@gmail.com) 

# DC Properties
*dataset ```DC Properties``` di download langsung dari website* https://www.kaggle.com/christophercorrea/dc-residential-properties

## WORKFLOW PROCESS

![Workflow](https://github.com/PurwadhikaDev/DataHeroes_JC_DS_AH_6_FinalProject/blob/main/img/FP-DCProp-Workflow.jpeg)


## 1. BUSINESS PROBLEMS

Bisnis Properti merupakan salah satu bisnis yang dengan resiko yang cukup tinggi. Ketika membeli suatu properti untuk dijual dalam jangka waktu yang cepat, para Pebisnis haruslah mengetahui Harga Wajar dari Properti yang akan di beli.
Harga wajar tersebut pada dasarnya dapat di diketahui apabila Pebisnis mempunyai:
1. Prediksi Harga
2. Pemetaan Resiko 
3. Tren Harga

Dengan Machine Learning, Pebisnis dapat dengan mudah mengetahi 3 hal di atas. sehingga Pebisnis dapat membeli suatu rumah dengan harga Wajar, dan dapat menjualnya dengan Profit maksimal. serta dengan adanya ML ini,  Pebisnis dapat membuat suatu Start Up dengan Bisnis Model : `AYH Inc` (Acquisition your House). `AYH Inc` memberikan alternatif solusi bagi pemilik rumah untuk menjual rumahnya dengan cepat dan mudah. Setelah properti tersebut lolos kualifikasi, dengan ML ini Peusahaan dapat memberikan penawaran harga  akan menawarkan properti Anda kepada database Tenant `AYH Inc`.

Dengan adanya ML ini, akan meminimalisih resiko dari setiap rumah yang akan dibeli oleh `AYH Inc`.

Sample Data yang yang dipakai adalah `DC Properties`, yaitu data rumah yang berada di Wasington DC. 

## 2. GOALS SETTINGS

- Mendapatkan analisis data DC Properties untuk mengetahui Prediksi Harga Properties di Wasington DC
- Mendapatkan analisis tentang pemetaan resiko dari harga jual di setiap wilayah
- Mengetahui trend harga rumah

OUTPUT
- Harga Wajar       = ```Hasil regresi``` x ```Hasil cluster resiko```
- Harga Jual Rumah  = ```Hasil Regresi``` x ```Tren Harga Rumah```

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



* `Regresi` 
    - Mendapatkan analisis data DC Properties untuk mengetahui Prediksi Harga Properties di Wasington DC
        1.	OLS
        2.	Linear Regression
        3.	DecisionTreeRegressor
        4.	Random Forest
        5.	Adaboost Decision Tree
        6.	XGB Regressor
        7.	KNN
        8.	SVM
        9.	Ridge

*  `Clustering`
    - menghitung resiko prosesntase harga jual (dipilih fitur yang kita perkirakan menjadi resiko menggunakan bisnis understanding)
    - Mendapatkan analisis tentang pemetaan resiko dari harga jual di setiap wilayah
        1.	KMeans
        2.	Agglomerative
        3.	Gaussian mixture model (GMM)


* `Time series` 
    - Mengetahui trend harga rumah
        1.	Dekomposisi Tren

  
