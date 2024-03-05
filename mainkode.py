#%%
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
import unicodedata as ud

##################################### TASK 1 ##################################

#%% Import price and prosumer data
df_prices, df_pro = LoadData()
df_prices = PricesDK(df_prices)

#%% Preserve only relevant cols
Yearly = df_prices.groupby(df_prices["HourDK"].dt.year)["SpotPriceDKK"].mean().reset_index()
Yearly = Yearly.rename(columns={'HourDK': 'Year'})

#%% Plot evolution of DK2 price
plt.figure()
plt.bar(Yearly["Year"],Yearly["SpotPriceDKK"])
plt.xlabel("Year")
plt.ylabel("Price [DKK/MWh]")
plt.title("Evolution of DK2 spot prices")

# 1.2

#%% Plot hourly avg. spot price
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

hourly = df_prices.groupby([df_prices["HourDK"].dt.hour])["SpotPriceDKK"].mean().reset_index()
hourly = hourly.rename(columns={'HourDK': 'Hour'})
daily = df_prices.groupby([df_prices["HourDK"].dt.day])["SpotPriceDKK"].mean().reset_index()
daily = daily.rename(columns={'HourDK': 'Day'})

df2019=df_prices.loc[df_prices['HourDK'].dt.year == 2019]
daily2019 = df2019.groupby([df2019["HourDK"].dt.day])["SpotPriceDKK"].mean().reset_index()
hourly2019 = df2019.groupby([df2019["HourDK"].dt.hour])["SpotPriceDKK"].mean().reset_index()
hourly2019 = hourly2019.rename(columns={'HourDK': 'Hour'})

df2020=df_prices.loc[df_prices['HourDK'].dt.year == 2020]
daily2020 = df2020.groupby([df2020["HourDK"].dt.day])["SpotPriceDKK"].mean().reset_index()
hourly2020 = df2020.groupby([df2020["HourDK"].dt.hour])["SpotPriceDKK"].mean().reset_index()
hourly2020 = hourly2020.rename(columns={'HourDK': 'Hour'})

df2021=df_prices.loc[df_prices['HourDK'].dt.year == 2021]
daily2021 = df2021.groupby([df2021["HourDK"].dt.day])["SpotPriceDKK"].mean().reset_index() 
hourly2021 = df2021.groupby([df2021["HourDK"].dt.hour])["SpotPriceDKK"].mean().reset_index()
hourly2021 = hourly2021.rename(columns={'HourDK': 'Hour'})

df2022=df_prices.loc[df_prices['HourDK'].dt.year == 2022]
daily2022 = df2022.groupby([df2022["HourDK"].dt.day])["SpotPriceDKK"].mean().reset_index()
hourly2022 = df2022.groupby([df2022["HourDK"].dt.hour])["SpotPriceDKK"].mean().reset_index()
hourly2022 = hourly2022.rename(columns={'HourDK': 'Hour'})

df2023=df_prices.loc[df_prices['HourDK'].dt.year == 2023]
daily2023 = df2023.groupby([df2023["HourDK"].dt.day])["SpotPriceDKK"].mean().reset_index()
hourly2023 = df2023.groupby([df2023["HourDK"].dt.hour])["SpotPriceDKK"].mean().reset_index()
hourly2023 = hourly2023.rename(columns={'HourDK': 'Hour'})

axs[0, 0].set_title('2018')
axs[0, 0].bar(hourly['Hour'], hourly['SpotPriceDKK'])
axs[0, 0].set_xlabel('Hour')
axs[0, 0].set_ylabel('Average Spot Price [DKK/MWh]')
axs[0, 0].set_ylim([0, max(hourly2022['SpotPriceDKK'])+100])

axs[0, 1].set_title('2019')
axs[0, 1].bar(hourly2019['Hour'], hourly2019['SpotPriceDKK'])
axs[0, 1].set_xlabel('Hour')
axs[0, 1].set_ylabel('Average Spot Price [DKK/MWh]')
axs[0, 1].set_ylim([0, max(hourly2022['SpotPriceDKK'])+100])

axs[0, 2].set_title('2020')
axs[0, 2].bar(hourly2020['Hour'], hourly2020['SpotPriceDKK'])
axs[0, 2].set_xlabel('Hour')
axs[0, 2].set_ylabel('Average Spot Price [DKK/MWh]')
axs[0, 2].set_ylim([0, max(hourly2022['SpotPriceDKK'])+100])

axs[1, 0].set_title('2021')
axs[1, 0].bar(hourly2021['Hour'], hourly2021['SpotPriceDKK'])
axs[1, 0].set_xlabel('Hour')
axs[1, 0].set_ylabel('Average Spot Price [DKK/MWh]')
axs[1, 0].set_ylim([0, max(hourly2022['SpotPriceDKK'])+100])

axs[1, 1].set_title('2022')
axs[1, 1].bar(hourly2022['Hour'], hourly2022['SpotPriceDKK'])
axs[1, 1].set_xlabel('Hour')
axs[1, 1].set_ylabel('Average Spot Price [DKK/MWh]')
axs[1, 1].set_ylim([0, max(hourly2022['SpotPriceDKK'])+100])

axs[1, 2].set_title('2023')
axs[1, 2].bar(hourly2023['Hour'], hourly2023['SpotPriceDKK'])
axs[1, 2].set_xlabel('Hour')
axs[1, 2].set_ylabel('Average Spot Price [DKK/MWh]')
axs[1, 2].set_ylim([0, max(hourly2022['SpotPriceDKK'])+100])

fig.suptitle('Hourly average spot price', size = 24)

plt.tight_layout()
plt.show()

daily_avg_prices = df_prices.groupby(df_prices["HourDK"].dt.date)["Buy"].mean().reset_index()

##################################### TASK 2 ##################################

# 2.1

#%% Parameters and arrays for optimization and aggregation
params = {
    'Pmax': 5/1000,
    'n_c': 0.99,
    'n_d': 0.99,
    'Cmax': 10/1000
}
params['C_0'] = 0.1 * params['Cmax']
params['C_n'] = 0.5 * params['Cmax']

yearprf = []
dailyprofit = []

#%% Loop through each day accounting for leap years, optimize, present yearly
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
            profitOpt, p_cOpt, p_dOpt, XOpt, dailyprof = Optimizer(params, p)
            prof.append(profitOpt)
            dailyprofit.append(profitOpt)
    profsum = sum(prof)
    print("Profit for " + str(year) + ": " + str(profsum) + " DKK.")
    yearprf.append(profsum)
    
#%% Plot yearly profits
plt.figure()
plt.bar(Yearly["Year"],yearprf)
plt.xlabel("Year")
plt.ylabel("Profit [DKK]")
plt.title("Yearly profit from 2019 to 2023")
plt.show()
plt.close()

# 2.2

#%% Plot daily profits
plt.figure()
plt.bar(daily_avg_prices["HourDK"],dailyprofit)
plt.bar(daily_avg_prices["HourDK"],daily_avg_prices["Buy"]/1000)
plt.xlabel("Year")
plt.ylabel("DKK")
plt.legend(["Profit in [DKK]", "Price [DKK/kWh]"])
plt.title("Daily profit and price from 2019 to 2023")
plt.show()
plt.close()

#%% Calculate correlation btw. profit and spot price
correlation = np.corrcoef(dailyprofit, daily_avg_prices["Buy"])
print("The correlation between the daily profit and the price is: " + str(round(correlation[0,1], 2)))

# 2.3 

#%% Parameters and arrays for optimization and aggregation
params = {
    'Pmax': 5/1000,
    'n_c': 0.95,
    'n_d': 0.95,
    'Cmax': 10/1000
}
params['C_0'] = 0.1 * params['Cmax']
params['C_n'] = 0.5 * params['Cmax']

yearprf95 = []
dailyprofit95 = []

#%% Loop through each day accounting for leap years, optimize
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
            dailyprofit95.append(profitOpt)
    profsum95 = sum(prof)
    yearprf95.append(profsum95)

#%% Parameters and arrays for optimization and aggregation
params = {
    'Pmax': 5/1000,
    'n_c': 0.90,
    'n_d': 0.90,
    'Cmax': 10/1000
}
params['C_0'] = 0.1 * params['Cmax']
params['C_n'] = 0.5 * params['Cmax']

yearprf90 = []
dailyprofit90 = []

#%% Loop through each day accounting for leap years, optimize
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
            dailyprofit90.append(profitOpt)
    profsum90 = sum(prof)
    yearprf90.append(profsum90)

#%% Plot yearly profits with different efficiencies
plt.figure()
plt.bar(Yearly["Year"],yearprf)
plt.bar(Yearly["Year"],yearprf95)
plt.bar(Yearly["Year"],yearprf90)
plt.xlabel("Year")
plt.ylabel("Profit [DKK]")
plt.title("Yearly profit from 2019 to 2023")
plt.legend(["\u03B7 = 0.99", "\u03B7 = 0.95", "\u03B7 = 0.90"])
plt.show()

##################################### TASK 3 ##################################

#%% Convert buy and sell to DKK/kWh
df_prices["Buy"] = df_prices["Buy"] / 1000
df_prices["Sell"] = df_prices["Sell"] / 1000

# 3.1

df_pro['consumer_cost'] = df_pro['Load'] * df_prices['Buy']

df_pro["Month"] = df_pro["HourDK"].dt.month
df_pro["Year"] = df_pro["HourDK"].dt.year
yearly_consumer_cost = df_pro.groupby('Year')['consumer_cost'].sum().reset_index()
print("price for load for each year exatly: \n", yearly_consumer_cost)

# Yearly estimate of load
df_year_load = df_pro.groupby(df_pro["HourDK"].dt.year)["Load"].sum().reset_index()
print("\n Total load for each year: \n", df_year_load)

df_year_buy = df_prices.loc[df_prices["HourDK"].dt.year.isin([2022, 2023])].groupby(df_prices["HourDK"].dt.year)["Buy"].mean().reset_index()

print("\n Average buying for each year: \n", df_year_buy)

load_price_year = df_year_buy["Buy"] * df_year_load["Load"]

print("\n price for load for each year using avage price: \n", load_price_year)

# 3.2

Net = Netting(df_pro, df_prices)

# Consumer cost is a expense but is a positive number, while profit is a gain and is a negative number
# so we add them together to get the yearly benefit of the system in terms of savings


print("yearly benefit of the system")
print(yearly_consumer_cost["consumer_cost"] + Net["Profit"])

df_pro["savings"] = df_pro["Profit"] + df_pro["consumer_cost"]






#%% 

df_scatter = df_pro.groupby(df_pro["HourDK"].dt.date)["savings"].sum().reset_index()
regtime = np.arange(0, len(df_scatter["HourDK"]))

df_scatter["cumulative_savings"] = df_scatter["savings"].cumsum()

#%% We load the csv files columns into numpy arrays
x =regtime
y = df_scatter["cumulative_savings"].values

#%% Fit linear regression model
LinReg = LinearRegression().fit(x.reshape(-1, 1), y)
LinPred = LinReg.predict(x.reshape(-1, 1))

x_pred = 7300
y_pred = LinReg.predict([[x_pred]])
print("Predicted y for x =", x_pred, "is", y_pred)

#%% Print the intercept and coefficient
print("Intercept (beta0):", LinReg.intercept_)
print("Coefficient (beta1):", LinReg.coef_)

#%% Plot the scatter plot and regression line
plt.scatter(x, y, c='b', alpha=0.5, label='Data', s = 5)
plt.plot(x, LinPred, color='r', label='Regression Line')
plt.xlabel('Days in 2 years')
plt.ylabel('Cumulative savings [DKK]')
plt.title('Linear regression model')
plt.legend()
plt.show()




#%% plot the scatter plot

plt.scatter(df_scatter["HourDK"], df_scatter["savings"])
plt.xlabel("Months in 2 years")
plt.ylabel("Savings [DKK]")
plt.xticks(rotation=60)
plt.title("Savings over time")
plt.show()

# 3.3

#%% The params for this assignamet
params = {
    'Pmax': 5,
    'n_c': 0.95,
    'n_d': 0.95,
    'Cmax': 10
}
params['C_0'] = 0.1 * params['Cmax']
params['C_n'] = 0.5 * params['Cmax']

consuption_cost = []
yearly_consup_cost = []
total_cost = 0

#%% Loop through each day accounting for leap years, optimize, present yearly
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
            # Strings for buy and sell prices and PV and Load
            # Taken from hands on
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
    print("The cost for " + str(year) + " is equal to: " + str(total_cost) + " DKK.")
    yearly_consup_cost.append(total_cost)
    total_cost = 0

#%% Plot daily consumption cost
plt.figure()  
plt.plot(consuption_cost)
plt.xlabel("Days in 2 years")
plt.ylabel("Consumption cost [DKK]")
plt.title("Daily consumption cost")
plt.show()


print(yearly_consup_cost)

consuption_cost = np.array(consuption_cost)

consuption_cost_cum=consuption_cost.cumsum()

#%% Plot the cumulative consumption cost
plt.plot(consuption_cost_cum)
plt.xlabel("Days in 2 years")
plt.ylabel("Cumulative consumption cost [DKK]")
plt.title("Cumulative consumption cost")
plt.show()

print("benefit of the system of battery and solar panel over nothing in terms of savings")
print(yearly_consumer_cost["consumer_cost"]-yearly_consup_cost)

print("benefit of the system of battery and solar panel over PV in terms of savings")
print(yearly_consumer_cost["consumer_cost"]-yearly_consup_cost+Net["Profit"])



# %%
