import pandas as pd
import numpy as np
from sklearn import linear_model
import math

df = pd.DataFrame({'area':[2600,3000,3200,3600,4000],'bedrooms':[3,4,np.NaN,3,5],'age':[20,15,18,30,8],'price':[550000,565000,610000,5950000,760000]})
print(df)
nan_value = df.isna()
nan_columns = nan_value. any()
columns_with_nan = df. columns[nan_columns]. tolist()
print('columns with NaN are:', columns_with_nan)

median_bedrooms = math.floor(df.bedrooms.median())
df.bedrooms = df.bedrooms.fillna(median_bedrooms)
print(df)
reg = linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)
print('Linear Regression Coefficients are ', reg.coef_)
print('Linear Regression. Intercept is ',reg.intercept_)
print("Predicted price for [32000,3,18]:",reg.predict([[32000, 3, 18]]))
