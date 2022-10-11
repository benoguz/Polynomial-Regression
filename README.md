### *Dataset*
> Data is generated by using Quasiperiodic function and additive noise 

# Polynomial Regression
> Data is generated by using Quasiperiodic function and additive noise 

polynomial regression model for a single predictor, X, is:

![image](https://user-images.githubusercontent.com/29160749/195036041-6bd8b748-d6be-4988-8f42-432323651cec.png)


where h is called the degree of the polynomial. For lower degrees, the relationship has a specific name (i.e., h = 2 is called quadratic, h = 3 is called cubic, h = 4 is called quartic, and so on). Although this model allows for a nonlinear relationship between Y and X, polynomial regression is still considered linear regression since it is linear in the regression coefficients β1, β2,..., βh!

https://online.stat.psu.edu/stat462/node/158/#:~:text=Although%20this%20model%20allows%20for,.%20.%20.%20%2C%20%CE%B2%20h%20
https://stats.stackexchange.com/questions/92065/why-is-polynomial-regression-considered-a-special-case-of-multiple-linear-regres

# Trend and Seasonality Analysis
* Polynomial regression

Lineer Regression Assumption:
data = model + noise

noise is a kind of gaussian or normal distribution so we need to check the residual by using the normality test (Shapiro-Wilk) but first let's look at trend.

![image](https://user-images.githubusercontent.com/29160749/195036200-59dd575f-599d-47f9-a4d6-6c351981dcc6.png)

* Statsmodel
>> time series analysis seasonal decomposition 

>> Hodrick-Prescott trend, cycle filter

# Time - Frequency Analysis
> Emprical Wavelet Transfrom
