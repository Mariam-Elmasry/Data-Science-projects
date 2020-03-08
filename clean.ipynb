{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import os\n",
    "import pandas as pd\n",
    "from sklearn import tree\n",
    "from sklearn import preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Read US Census construction dataset\n",
    "\n",
    "os.chdir( '/Users/healey/Downloads' )\n",
    "\n",
    "# Drop the first row, it's an \"explanation\" row and doesn't contain data,\n",
    "# but since it has strings it forces every column's values to also be\n",
    "# strings, which is NOT what we want\n",
    "\n",
    "df = pd.read_csv( 'construction-census.csv', sep=',', skiprows=[ 1 ] )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "col = df[ 'PAYANN' ]    # Get PAYANN column\n",
    "\n",
    "# loc() is used for label-based indexing\n",
    "\n",
    "row = df.loc[ 3 ]    # Get row with label of '3'\n",
    "row = df.loc[ 2: 4 ]  # Get rows with labels of '2', '3', and '4'\n",
    "row = df.loc[ [ 0, 2, 4 ] ]    # Get rows with labels of '0', '2', and '4'\n",
    "\n",
    "# For columns, specify : as first argument to get all rows\n",
    "\n",
    "col = df.loc[ :, 'PAYANN' ]    # Get column with label of 'PAYANN'\n",
    "col = df.loc[ :, 'ESTAB': 'EMP' ]  # Get columns from 'ESTAB' to 'EMP'\n",
    "col = df.loc[ :, [ 'EMP', 'PAYANN', 'BENVOL'] ]    # Get columns 'EMP', 'PAYANN', 'BENVOL'\n",
    "\n",
    "# iloc() is used for absolute (integer) indexing\n",
    "\n",
    "row = df.iloc[ 2 ]    # Get 3rd row (remember, indexing is 0-based)\n",
    "row = df.iloc[ 0: 3 ]    # Get 1st thru 3rd rows\n",
    "row = df.iloc[ [ 0, 2, 4 ] ]    # Get 1st, 3rd, and 5th rows\n",
    "\n",
    "col = df.iloc[ :, 2 ]    # Get 3rd column\n",
    "col = df.iloc[ :, 2: 4 ]    # Get 3rd and 4th column\n",
    "col = df.iloc[ :, [0, 2, 4 ] ]    # Get 1st, 3rd, and 5th columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['GEO.id', 'GEO.id2', 'GEO.display-label', 'GEO.annotation.id', 'NAICS.id', 'NAICS.display-label', 'NAICS.annotation.id', 'YEAR.id', 'ESTAB', 'ESTAB_S', 'EMP', 'EMP_S', 'PAYANN', 'PAYANN_S', 'BENEFIT', 'BENEFIT_S', 'BENLGL', 'BENVOL', 'EMPSMCD', 'EMPSMCD_S', 'EMPQ1CW', 'EMPQ2CW', 'EMPQ3CW', 'EMPQ4CW', 'EMPSMOD', 'EMPSMOD_S', 'EMPQ1OC', 'EMPQ2OC', 'EMPQ3OC', 'EMPQ4OC', 'PAYANOC', 'PAYANOC_S', 'HOURS', 'HOURS_S', 'PAYANCW', 'PAYANCW_S', 'PAYQTR1', 'PAYQTR1_S', 'CSTCMT', 'CSTMPRT', 'CSTMPRT_S', 'CSTSCNT', 'CSTSCNT_S', 'CSTEFT', 'CSTEFT_S', 'CSTFUGT', 'CSTFUNG', 'CSTFUON', 'CSTFUOF', 'CSTFUOT', 'CSTELEC', 'CSTELEC_S', 'RCPTOT', 'RCPTOT_S', 'RCPCWRK', 'RCPCWRK_S', 'RCPCGDL', 'FEDDL', 'SLDL', 'PRIDL', 'RCPOTH', 'RCPOTH_S', 'RCPCCNDL', 'RCPNCW', 'RCPNCW_S', 'VALADD', 'VALADD_S', 'INVMATB', 'INVMATB_S', 'INVMATE', 'INVMATE_S', 'ESTCIN', 'ESTCIN_S', 'RCPCIN', 'RCPCIN_S', 'ESTCNO', 'ESTCNO_S', 'RCPCNO', 'RCPCNO_S', 'ESTCNR', 'ESTCNR_S', 'RCPCNR', 'RCPCNR_S', 'ASTBGN', 'CEXTOT', 'CEXTOT_S', 'ASRET', 'ASTEND', 'ASTEND_S', 'DPRTOT', 'RPTOT', 'RPBLD', 'RPBLD_S', 'RPMCH', 'RPMCH_S', 'PCHTT', 'PCHTT_S', 'PCHTEMP', 'PCHCMPQ', 'PCHEXSO', 'PCHDAPR', 'PCHCSVC', 'PCHRPR', 'PCHRFUS', 'PCHADVT', 'PCHPRTE', 'PCHTAX', 'PCHOEXP']\n"
     ]
    }
   ],
   "source": [
    "# Get list of column names\n",
    "\n",
    "col_nm = list( df )\n",
    "print col_nm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Rows: 1575  Columns: 108  Observations: 170100\n"
     ]
    }
   ],
   "source": [
    "# Get row and column counts, and number of observations\n",
    "\n",
    "shp = df.shape\n",
    "row = shp[ 0 ]\n",
    "col = shp[ 1 ]\n",
    "n = row * col\n",
    "print 'Rows:', row, ' Columns:', col, ' Observations:', n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "25905\n"
     ]
    }
   ],
   "source": [
    "# Count number of missing (NaN) values\n",
    "\n",
    "nan_array = df.isnull().values\n",
    "nan_list = nan_array.tolist()\n",
    "\n",
    "nan_n = 0\n",
    "for i in range( 0, len( nan_list ) ):\n",
    "    for j in range( 0, len( nan_list[ i ] ) ):\n",
    "        if nan_list[ i ][ j ] == True:\n",
    "            nan_n = nan_n + 1\n",
    "\n",
    "print nan_n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "25905\n"
     ]
    }
   ],
   "source": [
    "# Or, comperss the 2D list into a 1D list, count number of True values\n",
    "\n",
    "nan_list = [ val for sublist in nan_list for val in sublist ]\n",
    "nan_n = nan_list.count( True )\n",
    "print nan_n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "25905\n"
     ]
    }
   ],
   "source": [
    "# Or, rely on the fact that True=1, False=0 and sum the True/False value list\n",
    "\n",
    "nan_n = df.isnull().values.sum()\n",
    "print nan_n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "GEO.annotation.id has 1575 NaN values; Removing empty column GEO.annotation.id\n",
      "NAICS.annotation.id has 1575 NaN values; Removing empty column NAICS.annotation.id\n",
      "EMP has 38 NaN values; EMP_S has 38 NaN values; PAYANN has 57 NaN values; PAYANN_S has 57 NaN values; BENEFIT has 61 NaN values; BENEFIT_S has 61 NaN values; BENLGL has 91 NaN values; BENVOL has 89 NaN values; EMPSMCD has 78 NaN values; EMPSMCD_S has 81 NaN values; EMPQ1CW has 88 NaN values; EMPQ2CW has 71 NaN values; EMPQ3CW has 68 NaN values; EMPQ4CW has 68 NaN values; EMPSMOD has 78 NaN values; EMPSMOD_S has 79 NaN values; EMPQ1OC has 60 NaN values; EMPQ2OC has 62 NaN values; EMPQ3OC has 72 NaN values; EMPQ4OC has 70 NaN values; PAYANOC has 103 NaN values; PAYANOC_S has 104 NaN values; HOURS has 62 NaN values; HOURS_S has 65 NaN values; PAYANCW has 102 NaN values; PAYANCW_S has 105 NaN values; PAYQTR1 has 63 NaN values; PAYQTR1_S has 63 NaN values; CSTCMT has 165 NaN values; CSTMPRT has 122 NaN values; CSTMPRT_S has 122 NaN values; CSTSCNT has 212 NaN values; CSTSCNT_S has 224 NaN values; CSTEFT has 265 NaN values; CSTEFT_S has 265 NaN values; CSTFUGT has 204 NaN values; CSTFUNG has 479 NaN values; CSTFUON has 294 NaN values; CSTFUOF has 253 NaN values; CSTFUOT has 441 NaN values; CSTELEC has 385 NaN values; CSTELEC_S has 386 NaN values; RCPTOT has 424 NaN values; RCPTOT_S has 424 NaN values; RCPCWRK has 118 NaN values; RCPCWRK_S has 118 NaN values; RCPCGDL has 271 NaN values; FEDDL has 418 NaN values; SLDL has 454 NaN values; PRIDL has 264 NaN values; RCPOTH has 436 NaN values; RCPOTH_S has 637 NaN values; RCPCCNDL has 144 NaN values; RCPNCW has 173 NaN values; RCPNCW_S has 173 NaN values; VALADD has 398 NaN values; VALADD_S has 398 NaN values; INVMATB has 324 NaN values; INVMATB_S has 359 NaN values; INVMATE has 230 NaN values; INVMATE_S has 268 NaN values; ESTCIN_S has 80 NaN values; RCPCIN has 611 NaN values; RCPCIN_S has 691 NaN values; ESTCNO_S has 11 NaN values; RCPCNO has 404 NaN values; RCPCNO_S has 415 NaN values; ESTCNR_S has 525 NaN values; RCPCNR has 659 NaN values; RCPCNR_S has 1184 NaN values; ASTBGN has 222 NaN values; CEXTOT has 202 NaN values; CEXTOT_S has 243 NaN values; ASRET has 272 NaN values; ASTEND has 302 NaN values; ASTEND_S has 302 NaN values; DPRTOT has 66 NaN values; RPTOT has 138 NaN values; RPBLD has 232 NaN values; RPBLD_S has 272 NaN values; RPMCH has 222 NaN values; RPMCH_S has 272 NaN values; PCHTT has 76 NaN values; PCHTT_S has 76 NaN values; PCHTEMP has 484 NaN values; PCHCMPQ has 357 NaN values; PCHEXSO has 536 NaN values; PCHDAPR has 479 NaN values; PCHCSVC has 192 NaN values; PCHRPR has 231 NaN values; PCHRFUS has 303 NaN values; PCHADVT has 268 NaN values; PCHPRTE has 174 NaN values; PCHTAX has 185 NaN values; PCHOEXP has 187 NaN values;\n"
     ]
    }
   ],
   "source": [
    "# To copy a DataFrame, you cannot simply equate them (df_cp = df), this creates a\n",
    "# \"reference\" from df_cp and df to the same DataFrame; instead, use copy()\n",
    "\n",
    "df_cp = df.copy()\n",
    "col_nm = list( df_cp )\n",
    "\n",
    "# Remove any column that's entirely NaN\n",
    "\n",
    "for nm in col_nm:\n",
    "    nan_n = df_cp[ nm ].isnull().values.sum()\n",
    "    if nan_n > 0:\n",
    "        print nm, 'has', nan_n, 'NaN values;',\n",
    "        \n",
    "    # Remove column if it's completely empty\n",
    "    \n",
    "    if nan_n == row:\n",
    "        print 'Removing empty column', nm\n",
    "        del df_cp[ nm ]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0      2078\n",
      "1      1128\n",
      "2      4343\n",
      "3      3875\n",
      "4     20944\n",
      "5     50004\n",
      "6      8914\n",
      "7      6395\n",
      "8     10538\n",
      "9       326\n",
      "10    23572\n",
      "11     2652\n",
      "12     2996\n",
      "13     1671\n",
      "14      190\n",
      "...\n",
      "1560      NaN\n",
      "1561     1891\n",
      "1562      100\n",
      "1563     1530\n",
      "1564    10625\n",
      "1565     7999\n",
      "1566      593\n",
      "1567     1654\n",
      "1568      544\n",
      "1569      NaN\n",
      "1570      330\n",
      "1571      303\n",
      "1572      NaN\n",
      "1573     6173\n",
      "1574     1234\n",
      "Name: BENVOL, Length: 1575, dtype: float64\n",
      "0      2078\n",
      "1      1128\n",
      "2      4343\n",
      "3      3875\n",
      "4     20944\n",
      "5     50004\n",
      "6      8914\n",
      "7      6395\n",
      "8     10538\n",
      "9       326\n",
      "10    23572\n",
      "11     2652\n",
      "12     2996\n",
      "13     1671\n",
      "14      190\n",
      "...\n",
      "1560     1150.5\n",
      "1561     1891.0\n",
      "1562      100.0\n",
      "1563     1530.0\n",
      "1564    10625.0\n",
      "1565     7999.0\n",
      "1566      593.0\n",
      "1567     1654.0\n",
      "1568      544.0\n",
      "1569      437.0\n",
      "1570      330.0\n",
      "1571      303.0\n",
      "1572     3238.0\n",
      "1573     6173.0\n",
      "1574     1234.0\n",
      "Name: BENVOL, Length: 1575, dtype: float64\n"
     ]
    }
   ],
   "source": [
    "# NaN values can be imputed in many different ways\n",
    "\n",
    "# fillna() fills NaN values with a fixed constant\n",
    "\n",
    "df_imp = df.copy()\n",
    "df_imp = df_imp.fillna( 0 )    # Fill all NaN in DataFrame with 0\n",
    "\n",
    "# fillna() can also be used with method='pad' to fill NaN with preceding\n",
    "# non-NaN value, or with method='backfill' to fill NaN with succeeding\n",
    "# non-NaN value; note that this will not fix a column of all NaN\n",
    "\n",
    "df_imp = df.copy()\n",
    "df_imp = df_imp.fillna( method='pad' )\n",
    "df_imp = df_imp.fillna( method='backfill' )\n",
    "\n",
    "# interpolate() fills values by interpolating over one or more NaN, different\n",
    "# methods are available like linear, nearest, spline, etc.\n",
    "\n",
    "# Note that interpolate() will throw an error if a column is all NaN\n",
    "\n",
    "df_imp = df.copy()\n",
    "df_imp[ 'BENVOL' ] = df_imp[ 'BENVOL' ].interpolate()\n",
    "\n",
    "print df[ 'BENVOL' ]\n",
    "print df_imp[ 'BENVOL' ]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Column GEO.id contains string(s), label encoding...\n",
      "Column GEO.display-label contains string(s), label encoding...\n",
      "Column NAICS.display-label contains string(s), label encoding...\n",
      "0      2078\n",
      "1      1128\n",
      "2      4343\n",
      "3      3875\n",
      "4     20944\n",
      "5     50004\n",
      "6      8914\n",
      "7      6395\n",
      "8     10538\n",
      "9       326\n",
      "10    23572\n",
      "11     2652\n",
      "12     2996\n",
      "13     1671\n",
      "14      190\n",
      "...\n",
      "1560      NaN\n",
      "1561     1891\n",
      "1562      100\n",
      "1563     1530\n",
      "1564    10625\n",
      "1565     7999\n",
      "1566      593\n",
      "1567     1654\n",
      "1568      544\n",
      "1569      NaN\n",
      "1570      330\n",
      "1571      303\n",
      "1572      NaN\n",
      "1573     6173\n",
      "1574     1234\n",
      "Name: BENVOL, Length: 1575, dtype: float64\n",
      "0      2078\n",
      "1      1128\n",
      "2      4343\n",
      "3      3875\n",
      "4     20944\n",
      "5     50004\n",
      "6      8914\n",
      "7      6395\n",
      "8     10538\n",
      "9       326\n",
      "10    23572\n",
      "11     2652\n",
      "12     2996\n",
      "13     1671\n",
      "14      190\n",
      "...\n",
      "1560     6184\n",
      "1561     1891\n",
      "1562      100\n",
      "1563     1530\n",
      "1564    10625\n",
      "1565     7999\n",
      "1566      593\n",
      "1567     1654\n",
      "1568      544\n",
      "1569     6184\n",
      "1570      330\n",
      "1571      303\n",
      "1572     6184\n",
      "1573     6173\n",
      "1574     1234\n",
      "Name: BENVOL, Length: 1575, dtype: float64\n"
     ]
    }
   ],
   "source": [
    "# scikit-learn can also impute NaN with its Imputer class, replacing NaN with\n",
    "# mean, median, or most_frequent value for a column/axis or entire DataFrame\n",
    "\n",
    "# Imputer cannot handle strings, however, so we must first convert any columns\n",
    "# with string values to numeric representations using Label Encoder\n",
    "\n",
    "df_imp = df_cp.copy()\n",
    "col_nm = list( df_cp )\n",
    "\n",
    "le = { }    # Label encoders, one per string column, key is column label\n",
    "\n",
    "for nm in col_nm:\n",
    "    \n",
    "    # Try to check for any string value, this will throw an error if the column\n",
    "    # is all-numeric, so catch that with except and do nothing (pass)\n",
    "    \n",
    "    try:\n",
    "        if True in df_imp[ nm ].str.contains( '.+' ).values:\n",
    "            print 'Column', nm, 'contains string(s), label encoding...'\n",
    "            \n",
    "            le[ nm ] = preprocessing.LabelEncoder()\n",
    "            le[ nm ].fit( df_imp[ nm ] )\n",
    "            df_imp[ nm ] = pd.Series( le[ nm ].transform( df_imp[ nm ] ) )\n",
    "    except:\n",
    "        pass\n",
    "\n",
    "imp = preprocessing.Imputer( strategy='median' )\n",
    "imp.fit( df_imp )\n",
    "\n",
    "col_nm = list( df_imp )\n",
    "df_imp = pd.DataFrame( imp.transform( df_imp ), columns=col_nm )\n",
    "\n",
    "print df[ 'BENVOL' ]\n",
    "print df_imp[ 'BENVOL' ]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Column GEO.display-label contains string(s), label encoding...\n",
      "Column GEO.id contains string(s), label encoding...\n",
      "Column NAICS.display-label contains string(s), label encoding...\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,\n",
       "            max_features=None, max_leaf_nodes=None,\n",
       "            min_impurity_split=1e-07, min_samples_leaf=1,\n",
       "            min_samples_split=2, min_weight_fraction_leaf=0.0,\n",
       "            presort=False, random_state=None, splitter='best')"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# We can also try to impute NaN using classification, e.g. a tree classifier; let's\n",
    "# try to impute NaN in the BENVOL column\n",
    "\n",
    "# Start by creating a training set whose rows have no NaN\n",
    "\n",
    "train = pd.DataFrame()\n",
    "\n",
    "for i, row in df_cp.iterrows():\n",
    "    if row.isnull().values.sum() == 0:\n",
    "        train = train.append( row )\n",
    "        \n",
    "# LabelEncode any columns with strings, because the classifier doesn't\n",
    "# handle strings\n",
    "\n",
    "col_nm = list( train )\n",
    "\n",
    "for nm in col_nm:\n",
    "    try:\n",
    "        if True in train[ nm ].str.contains( '.+' ).values:\n",
    "            print 'Column', nm, 'contains string(s), label encoding...'\n",
    "            \n",
    "            le = preprocessing.LabelEncoder()\n",
    "            le.fit( train[ nm ] )\n",
    "            \n",
    "            # Because we've removed rows w/NaN from train when we created\n",
    "            # it, its indices aren't contiguous 0,1,2,... they're based on\n",
    "            # the rows we kept 1,2,8,12,19... so when we build the label\n",
    "            # encoded series, we MUST match the same index values\n",
    "            \n",
    "            train[ nm ] = pd.Series( le.transform( train[ nm ] ), index=train[ nm ].index )\n",
    "    except:\n",
    "        pass\n",
    "\n",
    "# Now, grab the BENVOL column as a \"target\" and remove it from train, so we can\n",
    "# train on all columns except BENVOL to estimate BENVOL\n",
    "\n",
    "tg = train[ 'BENVOL' ]\n",
    "del train[ 'BENVOL' ]\n",
    "\n",
    "# Discretize the continuous BENVOL values into bins\n",
    "\n",
    "tg = pd.cut( tg, 100 )\n",
    "\n",
    "# Create and fit a sci-learn tree classifier\n",
    "\n",
    "clf = tree.DecisionTreeClassifier()\n",
    "clf.fit( train, tg )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0      2078\n",
      "1      1128\n",
      "2      4343\n",
      "3      3875\n",
      "4     20944\n",
      "5     50004\n",
      "6      8914\n",
      "7      6395\n",
      "8     10538\n",
      "9       326\n",
      "10    23572\n",
      "11     2652\n",
      "12     2996\n",
      "13     1671\n",
      "14      190\n",
      "...\n",
      "1560      NaN\n",
      "1561     1891\n",
      "1562      100\n",
      "1563     1530\n",
      "1564    10625\n",
      "1565     7999\n",
      "1566      593\n",
      "1567     1654\n",
      "1568      544\n",
      "1569      NaN\n",
      "1570      330\n",
      "1571      303\n",
      "1572      NaN\n",
      "1573     6173\n",
      "1574     1234\n",
      "Name: BENVOL, Length: 1575, dtype: float64\n",
      "0      2078\n",
      "1      1128\n",
      "2      4343\n",
      "3      3875\n",
      "4     20944\n",
      "5     50004\n",
      "6      8914\n",
      "7      6395\n",
      "8     10538\n",
      "9       326\n",
      "10    23572\n",
      "11     2652\n",
      "12     2996\n",
      "13     1671\n",
      "14      190\n",
      "...\n",
      "1560     5694.128\n",
      "1561     1891.000\n",
      "1562      100.000\n",
      "1563     1530.000\n",
      "1564    10625.000\n",
      "1565     7999.000\n",
      "1566      593.000\n",
      "1567     1654.000\n",
      "1568      544.000\n",
      "1569     5694.128\n",
      "1570      330.000\n",
      "1571      303.000\n",
      "1572     5694.128\n",
      "1573     6173.000\n",
      "1574     1234.000\n",
      "Name: BENVOL, Length: 1575, dtype: float64\n"
     ]
    }
   ],
   "source": [
    "# Now use the tree classifier to impute NaN in the original BENVOL column\n",
    "\n",
    "tg = df[ 'BENVOL' ].copy()\n",
    "\n",
    "# Use df_imp, which has all NaN imputed, because the classifier can't\n",
    "# handle NaN\n",
    "\n",
    "df_known = df_imp.copy()\n",
    "del df_known[ 'BENVOL' ]\n",
    "\n",
    "# Now walk through all BENVOL entries, use the classifier to impute any NaN\n",
    "\n",
    "for i in range( 0, len( tg ) ):\n",
    "    if np.isnan( tg.iloc[ i ] ):\n",
    "        \n",
    "        # Reshape an n x 1 column into a 1 x n vector, since the classifier wants\n",
    "        # known values in this format\n",
    "        \n",
    "        vec = df_known.iloc[ i ].reshape( 1, -1 )\n",
    "        rng = clf.predict( vec )[ 0 ]\n",
    "        \n",
    "        # Predictions are a string '(lo, hi]', so we parse this to get the\n",
    "        # median value between lo and hi and use that as our prediction\n",
    "        \n",
    "        rng = rng[ 1: -1 ]\n",
    "        rng = rng.split( ', ' )\n",
    "        val = float( rng[ 0 ] ) + float( rng[ 1 ] ) / 2.0\n",
    "\n",
    "        tg.iloc[ i ] = val\n",
    "\n",
    "print df[ 'BENVOL' ]\n",
    "print tg"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "####### "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
