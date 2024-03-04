import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import numpy as np
from Functions import PricesDK
from Functions import LoadData
from Functions import Optimizer
from Functions import ProsumerOptimizer
from Functions import Netting
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
#%%

### Import price and prosumer data ###
df_prices, df_pro = LoadData()
df_prices = PricesDK(df_prices)

Yearly = df_prices.groupby(df_prices["HourDK"].dt.year)["SpotPriceDKK"].mean().reset_index()
Yearly = Yearly.rename(columns={'HourDK': 'Year'})

plt.figure()
plt.bar(Yearly["Year"],Yearly["SpotPriceDKK"])
plt.xlabel("year")
plt.ylabel("Price in DKK/mWh")
plt.title("Evolution of DK2 spot prices")



fig, axs = plt.subplots(2, 3, figsize=(15, 10))

hourly = df_prices.groupby([df_prices["HourDK"].dt.hour])["SpotPriceDKK"].mean().reset_index()
hourly = hourly.rename(columns={'HourDK': 'Hour'})
dayly = df_prices.groupby([df_prices["HourDK"].dt.day])["SpotPriceDKK"].mean().reset_index()
dayly = dayly.rename(columns={'HourDK': 'Day'})


df2019=df_prices.loc[df_prices['HourDK'].dt.year == 2019]

dayly2019 = df2019.groupby([df2019["HourDK"].dt.day])["SpotPriceDKK"].mean().reset_index()
hourly2019 = df2019.groupby([df2019["HourDK"].dt.hour])["SpotPriceDKK"].mean().reset_index()
hourly2019 = hourly2019.rename(columns={'HourDK': 'Hour'})

df2020=df_prices.loc[df_prices['HourDK'].dt.year == 2020]
dayly2020 = df2020.groupby([df2020["HourDK"].dt.day])["SpotPriceDKK"].mean().reset_index()
hourly2020 = df2020.groupby([df2020["HourDK"].dt.hour])["SpotPriceDKK"].mean().reset_index()
hourly2020 = hourly2020.rename(columns={'HourDK': 'Hour'})

df2021=df_prices.loc[df_prices['HourDK'].dt.year == 2021]
dayly2021 = df2021.groupby([df2021["HourDK"].dt.day])["SpotPriceDKK"].mean().reset_index() 
hourly2021 = df2021.groupby([df2021["HourDK"].dt.hour])["SpotPriceDKK"].mean().reset_index()
hourly2021 = hourly2021.rename(columns={'HourDK': 'Hour'})

df2022=df_prices.loc[df_prices['HourDK'].dt.year == 2022]
dayly2022 = df2022.groupby([df2022["HourDK"].dt.day])["SpotPriceDKK"].mean().reset_index()
hourly2022 = df2022.groupby([df2022["HourDK"].dt.hour])["SpotPriceDKK"].mean().reset_index()
hourly2022 = hourly2022.rename(columns={'HourDK': 'Hour'})

df2023=df_prices.loc[df_prices['HourDK'].dt.year == 2023]
dayly2023 = df2023.groupby([df2023["HourDK"].dt.day])["SpotPriceDKK"].mean().reset_index()
hourly2023 = df2023.groupby([df2023["HourDK"].dt.hour])["SpotPriceDKK"].mean().reset_index()
hourly2023 = hourly2023.rename(columns={'HourDK': 'Hour'})





axs[0, 0].bar(hourly['Hour'], hourly['SpotPriceDKK'])
axs[0, 0].set_xlabel('Hour')
axs[0, 0].set_ylabel('Average Spot Price (DKK/MWh)')
axs[0, 0].set_title('Hourly Average Spot Price 2019-2023')
axs[0, 0].set_ylim([0, max(hourly2022['SpotPriceDKK'])+100])

axs[0, 1].bar(hourly2019['Hour'], hourly2019['SpotPriceDKK'])
axs[0, 1].set_xlabel('Hour')
axs[0, 1].set_ylabel('Average Spot Price (DKK/MWh)')
axs[0, 1].set_title('Hourly Average Spot Price 2019')
axs[0, 1].set_ylim([0, max(hourly2022['SpotPriceDKK'])+100])

axs[0, 2].bar(hourly2020['Hour'], hourly2020['SpotPriceDKK'])
axs[0, 2].set_xlabel('Hour')
axs[0, 2].set_ylabel('Average Spot Price (DKK/MWh)')
axs[0, 2].set_title('Hourly Average Spot Price 2020')
axs[0, 2].set_ylim([0, max(hourly2022['SpotPriceDKK'])+100])

axs[1, 0].bar(hourly2021['Hour'], hourly2021['SpotPriceDKK'])
axs[1, 0].set_xlabel('Hour')
axs[1, 0].set_ylabel('Average Spot Price (DKK/MWh)')
axs[1, 0].set_title('Hourly Average Spot Price 2021')
axs[1, 0].set_ylim([0, max(hourly2022['SpotPriceDKK'])+100])

axs[1, 1].bar(hourly2022['Hour'], hourly2022['SpotPriceDKK'])
axs[1, 1].set_xlabel('Hour')
axs[1, 1].set_ylabel('Average Spot Price (DKK/MWh)')
axs[1, 1].set_title('Hourly Average Spot Price 2022')
axs[1, 1].set_ylim([0, max(hourly2022['SpotPriceDKK'])+100])

axs[1, 2].bar(hourly2023['Hour'], hourly2023['SpotPriceDKK'])
axs[1, 2].set_xlabel('Hour')
axs[1, 2].set_ylabel('Average Spot Price (DKK/MWh)')
axs[1, 2].set_title('Hourly Average Spot Price 2023')
axs[1, 2].set_ylim([0, max(hourly2022['SpotPriceDKK'])+100])


plt.tight_layout()
plt.show()


daily_avg_prices = df_prices.groupby(df_prices["HourDK"].dt.date)["Buy"].mean().reset_index()


##################################### TASK 2 ##################################
#%%
#2.1
params = {
    'Pmax': 5/1000,
    'n_c': 0.99,
    'n_d': 0.99,
    'Cmax': 10/1000
}
params['C_0'] = 0.1 * params['Cmax']
params['C_n'] = 0.5 * params['Cmax']


yearprf = []


#for day in range(len(df_prices["HourDK"])):
daylyprofit = []


for year in range(2019, 2024):
    profsum = 0
    prof = []
    
    for month in range(1, 13):
        if month == 2:
            if year % 4 == 0:
                d = 30
            else:
                d = 29
        elif month in [4, 6, 9, 11]:
            d = 31
        else:
            d = 32
        for day in range(1, d):
            t_s = pd.Timestamp(dt.datetime(year, month, day, 0, 0, 0))
            t_e = pd.Timestamp(dt.datetime(year, month, day, 23, 0, 0))
            p = df_prices.loc[(df_prices["HourDK"]>=t_s) & (df_prices["HourDK"]<=t_e),"Sell"].values
            profitOpt, p_cOpt, p_dOpt, XOpt, daylyprof = Optimizer(params, p)
            prof.append(profitOpt)
            daylyprofit.append(profitOpt)
    profsum = sum(prof)
    print("Profit for " + str(year) + ": " + str(profsum) + " DKK.")
    yearprf.append(profsum)
    
    
            
    
    


plt.figure()
plt.bar(Yearly["Year"],yearprf)
plt.xlabel("year")
plt.ylabel("profit in DKK")
plt.title("Yearly profit from 2019 to 2023")
plt.show()
plt.close()

#2.2

plt.figure()
plt.bar(daily_avg_prices["HourDK"],daylyprofit)
plt.bar(daily_avg_prices["HourDK"],daily_avg_prices["Buy"]/1000)
plt.xlabel("year")
plt.ylabel("DKK")
plt.legend(["Profit in dkk", "Price in DKK/kWh"])
plt.title("Dayly profit and Price from 2019 to 2023")
plt.show()
plt.close()

correlation = np.corrcoef(daylyprofit, daily_avg_prices["Buy"])
print("The correlation between the dayly profit and the price is: " + str(correlation[0,1]))



#2.3 

params = {
    'Pmax': 5/1000,
    'n_c': 0.95,
    'n_d': 0.95,
    'Cmax': 10/1000
}
params['C_0'] = 0.1 * params['Cmax']
params['C_n'] = 0.5 * params['Cmax']


yearprf95 = []


daylyprofit95 = []


for year in range(2019, 2024):
    profsum95 = 0
    prof = []
    for month in range(1, 13):
        if month == 2:
            if year % 4 == 0:
                d = 30
            else:
                d = 29
        elif month in [4, 6, 9, 11]:
            d = 31
        else:
            d = 32
        for day in range(1, d):
            t_s = pd.Timestamp(dt.datetime(year, month, day, 0, 0, 0))
            t_e = pd.Timestamp(dt.datetime(year, month, day, 23, 0, 0))
            p = df_prices.loc[(df_prices["HourDK"]>=t_s) & (df_prices["HourDK"]<=t_e),"Sell"].values
            profitOpt, p_cOpt, p_dOpt, XOpt, dayprof = Optimizer(params, p)
            prof.append(profitOpt)
            daylyprofit95.append(profitOpt)
    profsum95 = sum(prof)
    #print("Profit for " + str(year) + ": " + str(profsum) + " DKK.")
    yearprf95.append(profsum95)


params = {
    'Pmax': 5/1000,
    'n_c': 0.90,
    'n_d': 0.90,
    'Cmax': 10/1000
}
params['C_0'] = 0.1 * params['Cmax']
params['C_n'] = 0.5 * params['Cmax']


yearprf90 = []


daylyprofit90 = []


for year in range(2019, 2024):
    profsum90 = 0
    prof = []

    for month in range(1, 13):
        if month == 2:
            if year % 4 == 0:
                d = 30
            else:
                d = 29
        elif month in [4, 6, 9, 11]:
            d = 31
        else:
            d = 32
        for day in range(1, d):
            t_s = pd.Timestamp(dt.datetime(year, month, day, 0, 0, 0))
            t_e = pd.Timestamp(dt.datetime(year, month, day, 23, 0, 0))
            p = df_prices.loc[(df_prices["HourDK"]>=t_s) & (df_prices["HourDK"]<=t_e),"Sell"].values
            profitOpt, p_cOpt, p_dOpt, XOpt, dayprof = Optimizer(params, p)
            prof.append(profitOpt)
            daylyprofit90.append(profitOpt)
    profsum90 = sum(prof)
    #print("Profit for " + str(year) + ": " + str(profsum) + " DKK.")
    yearprf90.append(profsum90)

    
plt.figure()
plt.bar(Yearly["Year"],yearprf)
plt.bar(Yearly["Year"],yearprf95)
plt.bar(Yearly["Year"],yearprf90)
plt.xlabel("year")
plt.ylabel("profit in DKK")
plt.title("yerly profit from 2019 to 2023")
plt.legend(["Efficiency 0.99", "Efficiency 0.95", "Efficiency 0.90"])
plt.show()




#%%

### task 3 ###

"""
# covert from kw to mw
df_pro["Load"] = df_pro["Load"] / 1000
df_pro["PV"] = df_pro["PV"] / 1000
"""
#3.1
df_pro['consumer_cost'] = df_pro['Load'] * df_prices['Buy']

df_pro["Month"] = df_pro["HourDK"].dt.month
df_pro["Year"] = df_pro["HourDK"].dt.year
yearly_consumer_cost = df_pro.groupby('Year')['consumer_cost'].sum().reset_index()
print(yearly_consumer_cost)


# årligt esteimat for forbrug
df_year_load = df_pro.groupby(df_pro["HourDK"].dt.year)["Load"].sum().reset_index()
print( "total load for each year: \n", df_year_load)


df_year_buy = df_prices.loc[df_prices["HourDK"].dt.year.isin([2022, 2023])].groupby(df_prices["HourDK"].dt.year)["Buy"].mean().reset_index()

print("avage price for buy for each year: \n", df_year_buy)

load_price_year = df_year_buy["Buy"] * df_year_load["Load"]

print("total price for load for each year: \n", load_price_year)


#3.2

Net = Netting(df_pro, df_prices)

#consumer cost er negativ 
print(yearly_consumer_cost["consumer_cost"])
print(Net["Profit"])
print(yearly_consumer_cost["consumer_cost"]+Net["Profit"])


df_pro["savings"] = df_pro["Profit"] + df_pro["consumer_cost"]



df_scatter = df_pro.groupby(df_pro["HourDK"].dt.date)["savings"].sum().reset_index()


#%% 

regtime = np.arange(0, len(df_scatter["HourDK"]))

df_scatter["cumulative_savings"] = df_scatter["savings"].cumsum()

#df_scatter["HourDKstring"] = df_scatter["HourDK"].to_numpy().astype(str)
# We load the csv files columns into numpy arrays
x =regtime
y = df_scatter["cumulative_savings"].values

# Create polynomial features
poly = PolynomialFeatures(degree=3, include_bias=False).fit_transform(x.reshape(-1, 1))

# Fit linear regression model on polynomial features
LinReg_poly = LinearRegression().fit(poly, y)
LinPred_poly = LinReg_poly.predict(poly)

# Print the residuals, intercept, and coefficients
print("Intercept (beta0):", LinReg_poly.intercept_)
print("Coefficients (beta1, beta2, beta3):", LinReg_poly.coef_)

# Plot the scatter plot and regression line
plt.scatter(x, y, c='b', alpha=0.5, label='Data')
plt.plot(x, LinPred_poly, color='r', label='Regression Line')

# Set the labels for x-, y-axes & title
plt.xlabel('Descriptive Variable (x)')
plt.ylabel('Predicted Variable (y)')
plt.title('3rd Degree Polynomial Regression Model')

# Show the legends & plot
plt.legend()
plt.show()




plt.scatter(df_scatter["HourDK"], df_scatter["savings"])
plt.xlabel("Time")
plt.ylabel("Savings")
plt.title("Savings over Time")
plt.show()


#%%
# 3.3?
# the params for this assignamet


#convert buy and sell to dkk/kwh
df_prices["Buy"] = df_prices["Buy"] / 1000
df_prices["Sell"] = df_prices["Sell"] / 1000


params = {
    'Pmax': 5,
    'n_c': 0.95,
    'n_d': 0.95,
    'Cmax': 10
}
params['C_0'] = 0.1 * params['Cmax']
params['C_n'] = 0.5 * params['Cmax']

consuption_cost = []
total_cost = 0

for year in range(2022, 2024):
    for month in range(1, 13):
        if month == 2:
            if year % 4 == 0:
                d = 30
            else:
                d = 29
        elif month in [4, 6, 9, 11]:
            d = 31
        else:
            d = 32
        for day in range(1, d):
            # strings for buy and sell prices and PV and Load
            # taken from hands on
            t_s = pd.Timestamp(dt.datetime(year, month, day, 0, 0, 0))
            t_e = pd.Timestamp(dt.datetime(year, month, day, 23, 0, 0))
            l_b = df_prices.loc[(df_prices["HourDK"]>=t_s) & (df_prices["HourDK"]<=t_e),"Buy"].values
            l_s = df_prices.loc[(df_prices["HourDK"]>=t_s) & (df_prices["HourDK"]<=t_e),"Sell"].values
            p_PV = df_pro.loc[(df_pro["HourDK"]>=t_s) & (df_pro["HourDK"]<=t_e),"PV"].values
            p_L = df_pro.loc[(df_pro["HourDK"]>=t_s) & (df_pro["HourDK"]<=t_e),"Load"].values
            # Call the ProsumerOptimizer function
            costOpt, p_cOpt, p_dOpt, p_bOpt, p_sOpt, XOpt = ProsumerOptimizer(params, l_b, l_s, p_PV, p_L)
            consuption_cost.append(costOpt)
            total_cost += costOpt
            #print(" costOpt" + str(costOpt) + "+p_cOpt" + str(p_cOpt) + "+p_dOpt" + str(p_dOpt) + "+p_bOpt" + str(p_bOpt) + "+p_sOpt" + str(p_sOpt) + "+XOpt" + str(XOpt))
    print("The cost for " + str(year) + " is equal to:" + str(total_cost) + " DKK.")
    total_cost = 0


plt.figure()  
plt.plot(consuption_cost)
plt.xlabel("day")
plt.ylabel("Consumption Cost (DKK)")
plt.title("Monthly Consumption Cost")
plt.show()

#%%
