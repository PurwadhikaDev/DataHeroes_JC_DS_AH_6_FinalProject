{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# DataHeroes_JC_DS_AH_6_FinalProject\n",
    "\n",
    "by  \n",
    "- Bayu Andrianto Wirawan (bay.psil@gmail.com)\n",
    "- Shafwan Hanif (shafwanh17@gmail.com) "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# DC Properties\n",
    "*dataset di download langsung dari website ```DC Properties``` di* https://www.kaggle.com/christophercorrea/dc-residential-properties\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## WORKFLOW PROCESS\n",
    "\n",
    "![Work Flow](./img/FP-DCProp-Workflow.jpeg \"FP-DCProp-Workflow.jpeg\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. BUSINESS PROBLEMS\n",
    "\n",
    "Bisnis Properti merupakan salah satu bisnis yang dengan resiko yang cukup tinggi. Ketika membeli suatu properti untuk dijual dalam jangka waktu yang cepat, para Pebisnis haruslah mengetahui Harga Wajar dari Properti yang akan di beli.\n",
    "Harga wajar tersebut pada dasarnya dapat di diketahui apabila Pebisnis mempunyai:\n",
    "1. Prediksi Harga\n",
    "2. Pemetaan Resiko \n",
    "3. Tren Harga\n",
    "\n",
    "Dengan Machine Learning, Pebisnis dapat dengan mudah mengetahi 3 hal di atas. sehingga Pebisnis dapat membeli suatu rumah dengan harga Wajar, dan dapat menjualnya dengan Profit maksimal. serta dengan adanya ML ini,  Pebisnis dapat membuat suatu Start Up dengan Bisnis Model : `AYH Inc` (Acquisition your House). `AYH Inc` memberikan alternatif solusi bagi pemilik rumah untuk menjual rumahnya dengan cepat dan mudah. Setelah properti tersebut lolos kualifikasi, dengan ML ini Peusahaan dapat memberikan penawaran harga  akan menawarkan properti Anda kepada database Tenant `AYH Inc`.\n",
    "\n",
    "Dengan adanya ML ini, akan meminimalisih resiko dari setiap rumah yang akan dibeli oleh `AYH Inc`.\n",
    "\n",
    "Sample Data yang yang dipakai adalah `DC Properties`, yaitu data rumah yang berada di Wasington DC. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. GOALS SETTINGS\n",
    "\n",
    "- Mendapatkan analisis data DC Properties untuk mengetahui Prediksi Harga Properties di Wasington DC\n",
    "- Mendapatkan analisis tentang pemetaan resiko dari harga jual di setiap wilayah\n",
    "- Mengetahui trend harga rumah\n",
    "\n",
    "OUTPUT\n",
    "- Harga Wajar       = ```Hasil regresi``` x ```Hasil cluster resiko```\n",
    "- Harga Jual Rumah  = ```Hasil Regresi``` x ```Tren Harga Rumah```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. DATA UNDERSTANDING\n",
    "\n",
    "Dalam tahap EDA untuk Regressi ini, ada beberapa step dan output yang di harapakan, sehingga dapat memberika hasil maksimal dalam melakukan machine learning. \n",
    "\n",
    "Dalam data `DC_Properties` akan di lakukan EDA dengan 2 kelompok data `SOURCE`, karena terdapat 2 tipe Properties. Yaitu:\n",
    "\n",
    "`RESIDENSIAL` vs `CONDOMINIUM`\n",
    "\n",
    "\n",
    "![Data Perbandingan](./img/ResvsCon.png \"ResvsCon.png\")\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Data Dictionary\n",
    "\n",
    "* \tBATHRM\t: \t\"Number of Full Bathrooms\"\n",
    "* \tHF_BATHRM\t: \tNumber of Half Bathrooms (no bathtub or shower)\n",
    "* \tHEAT\t: \tHeating\n",
    "* \tAC\t: \tCooling\n",
    "* \tNUM_UNITS\t: \tNumber of Units\n",
    "* \tROOMS\t: \tNumber of Rooms\n",
    "* \tBEDRM\t: \tNumber of Bedrooms\n",
    "* \tAYB\t: \tThe earliest time the main portion of the building was built\n",
    "* \tYR_RMDL\t: \tYear structure was remodeled\n",
    "* \tEYB\t: \tThe year an improvement was built more recent than actual year built\n",
    "* \tSTORIES\t: \tNumber of stories in primary dwelling\n",
    "* \tSALEDATE\t: \tDate of most recent sale\n",
    "* \tPRICE\t: \tPrice of most recent sale\n",
    "* \tQUALIFIED\t: \tQualified\n",
    "* \tSALE_NUM\t: \tSale Number\n",
    "* \tGBA\t: \tGross building area in square feet\n",
    "* \tBLDG_NUM\t: \tBuilding Number on Property\n",
    "* \tSTYLE\t: \tStyle\n",
    "* \tSTRUCT\t: \tStructure\n",
    "* \tGRADE\t: \tGrade\n",
    "* \tCNDTN\t: \tCondition\n",
    "* \tEXTWALL\t: \tExtrerior wall\n",
    "* \tROOF\t: \tRoof type\n",
    "* \tINTWALL\t: \tInterior wall\n",
    "* \tKITCHENS\t: \tNumber of kitchens\n",
    "* \tFIREPLACES\t: \tNumber of fireplaces\n",
    "* \tUSECODE\t: \tProperty use code\n",
    "* \tLANDAREA\t: \tLand area of property in square feet\n",
    "* \tGIS_LAST_MOD_DTTM\t: \tLast Modified Date\n",
    "* \tSOURCE\t: \tRaw Data Source\n",
    "* \tCMPLX_NUM\t: \tComplex number\n",
    "* \tLIVING_GBA\t: \tGross building area in square feet\n",
    "* \tFULLADDRESS\t: \tFull Street Address\n",
    "* \tCITY\t: \tCity\n",
    "* \tSTATE\t: \tState\n",
    "* \tZIPCODE\t: \tZip Code\n",
    "* \tNATIONALGRID\t: \tAddress location national grid coordinate spatial address\n",
    "* \tLATITUDE\t: \tLatitude\n",
    "* \tLONGITUDE\t: \tLongitude\n",
    "* \tASSESSMENT_NBHD\t: \tNeighborhood ID\n",
    "* \tASSESSMENT_SUBNBHD\t: \tSubneighborhood ID\n",
    "* \tCENSUS_TRACT\t: \tCensus tract\n",
    "* \tCENSUS_BLOCK\t: \tCensus block\n",
    "* \tWARD\t: \t\"Ward (District is divided into eight wards, each with approximately 75,000 residents)\n",
    "\"\n",
    "* \tSQUARE\t: \tSquare (from SSL - (Square, Suffix, Lot))\n",
    "* \tX\t: \tLatitude\n",
    "* \tY\t: \tLongitude\n",
    "* \tQUADRANT\t: \tCity quadrant (NE,SE,SW,NW)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. DATA CLEANING, EDA\n",
    "Beberapa step yang akan di lakukan adalah:\n",
    "\n",
    "1. Load Dataset  \n",
    "2. Menghitung Missing data\n",
    "3. Data Prepocessing\n",
    "    - Drop Data\n",
    "    - Mengisi NaN dengan Data Lain\n",
    "    - Mengisi NaN dengan Median\n",
    "    - Mengisi NaN dengan Modus\n",
    "4. Feature Enginering\n",
    "5. Data Preprocessing\n",
    "    - Dummy Variable\n",
    "6. Normalized Price\n",
    "    - melakukakan Normalisai Price - Kolom Price ditransform menggunakan Kubik\n",
    "7. Output\n",
    "8. Data Nominal\n",
    "9. Data Visualization\n",
    "\n",
    "\n",
    "Output Data yang dihasilkan Adalah\n",
    "- Residensial\n",
    "    1. Residensial for Reggresion `('Data/DC_Prop_Residential_Regression.csv')`\n",
    "    2. Residensial for Clustering `('Data/DC_Prop_Residential1.csv')`\n",
    "    3. Residensial for Time Serie `('Data/DC_Prop_Residential2.csv')`\n",
    "- Condominium\n",
    "    1. Condominium for Reggresion `('Data/DC_Prop_Condominium_Regression.csv')`\n",
    "    2. Condominium for Clustering `('Data/DC_Prop_Condominium1.csv')`\n",
    "    3. Condominium for Time Serie `('Data/DC_Prop_Condominium2.csv')`"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. MACHINE LEARNING MODELLING\n",
    "\n",
    "\n",
    "\n",
    "* `Regresi` \n",
    "    - Mendapatkan analisis data DC Properties untuk mengetahui Prediksi Harga Properties di Wasington DC\n",
    "        1.\tOLS\n",
    "        2.\tAutoML 🡪 TPOT\n",
    "        3.\tLinear Regresion\n",
    "        4.\tRegularisasi (Ridge / Lasso)\n",
    "        5.\tKNN\n",
    "        6.\tSVM\n",
    "        7.\tRF\n",
    "        8.\tAdaboost\n",
    "\n",
    "*  `Clustering`\n",
    "    - menghitung resiko prosesntase harga jual (dipilih fitur yang kita perkirakan menjadi resiko menggunakan bisnis understanding)\n",
    "    - Mendapatkan analisis tentang pemetaan resiko dari harga jual di setiap wilayah\n",
    "        1.\tKMeans\n",
    "        2.\tAgglomerative\n",
    "        3.\tDBScan\n",
    "\n",
    "\n",
    "* `Time series` \n",
    "    - Mengetahui trend harga rumah\n",
    "        1.\tDekomposisi Tren\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "a) Analisa data: Siswa mampu menyampaikan insight (temuan) hasil analisis sesuai dengan masalah yang telah didefinisikan. Temuan yang disampaikan membantu menyelesaikan masalah yang ditetapkan di awal.\n",
    "\n",
    "\n",
    "b)\tAnalisa data: Siswa mampu `memilih plot yang tepat untuk memudahkan stakeholder` memahami insight dari data yang dianalisis.\n",
    "\n",
    "c)\tPemodelan: Siswa mampu `menjelaskan cara kerja (algoritma) model ML` yang digunakan.\n",
    "\n",
    "d)\tPemodelan: Siswa mampu `menjelaskan cara kerja evaluation metrics` yang digunakan, misalnya: Regresi (MAE, MSE, RMSE, MAPE, MPSE, MSLE, R-Squared), Klasifikasi (accuracy, recall, precision, F1 Score, ROC AUC, PR Score), dan lainnya.\n",
    "\n",
    "e)\tPemodelan: Siswa mampu memilih evaluation metrics dan mampu `menjelaskan alasan pemilihannya`, serta keterhubungan evaluation metrics dengan aspek bisnis. Bahkan dapat membuat metric baru tambahan sendiri bila memungkinkan dan diperlukan.\n",
    "\n",
    "f)\tPemodelan: Siswa memahami bagaimana `implementasi` penggunaan model ML nantinya, seperti siapa yang akan pakai model ML-nya, dan kapan digunakannya.\n",
    "\n",
    "g)\tPemodelan: Siswa memahami kapan model ML bisa dipercaya dan kapan tidak bisa dipercaya (Explainable and Interpretable Model). Siswa harus memahami data-data seperti apa model bisa dipercaya (akurat) dan data-data seperti apa model masih belum dapat dipercaya (masih kurang akurat).\n"
   ]
  }
 ],
 "metadata": {
  "interpreter": {
   "hash": "c717e48f47dbf3882205ebaefec1a2b4f1b354b496a24c57e7aae83aa2cbf878"
  },
  "kernelspec": {
   "display_name": "Python 3.9.6 64-bit",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.9.6"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
