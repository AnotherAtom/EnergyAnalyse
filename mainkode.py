import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

from Functions import PricesDK
from Functions import LoadData
#from Functions import Netting
from Functions import Optimizer
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
axs[0, 0].set_ylim([0, max(hourly2022['SpotPriceDKK'])])

axs[0, 1].bar(hourly2019['Hour'], hourly2019['SpotPriceDKK'])
axs[0, 1].set_xlabel('Hour')
axs[0, 1].set_ylabel('Average Spot Price (DKK/MWh)')
axs[0, 1].set_title('Hourly Average Spot Price 2019')
axs[0, 1].set_ylim([0, max(hourly2022['SpotPriceDKK'])])

axs[0, 2].bar(hourly2020['Hour'], hourly2020['SpotPriceDKK'])
axs[0, 2].set_xlabel('Hour')
axs[0, 2].set_ylabel('Average Spot Price (DKK/MWh)')
axs[0, 2].set_title('Hourly Average Spot Price 2020')
axs[0, 2].set_ylim([0, max(hourly2022['SpotPriceDKK'])])

axs[1, 0].bar(hourly2021['Hour'], hourly2021['SpotPriceDKK'])
axs[1, 0].set_xlabel('Hour')
axs[1, 0].set_ylabel('Average Spot Price (DKK/MWh)')
axs[1, 0].set_title('Hourly Average Spot Price 2021')
axs[1, 0].set_ylim([0, max(hourly2022['SpotPriceDKK'])])

axs[1, 1].bar(hourly2022['Hour'], hourly2022['SpotPriceDKK'])
axs[1, 1].set_xlabel('Hour')
axs[1, 1].set_ylabel('Average Spot Price (DKK/MWh)')
axs[1, 1].set_title('Hourly Average Spot Price 2022')
axs[1, 1].set_ylim([0, max(hourly2022['SpotPriceDKK'])])

axs[1, 2].bar(hourly2023['Hour'], hourly2023['SpotPriceDKK'])
axs[1, 2].set_xlabel('Hour')
axs[1, 2].set_ylabel('Average Spot Price (DKK/MWh)')
axs[1, 2].set_title('Hourly Average Spot Price 2023')
axs[1, 2].set_ylim([0, max(hourly2022['SpotPriceDKK'])])


plt.tight_layout()
plt.show()


daily_avg_prices = df_prices.groupby(df_prices["HourDK"].dt.date)["SpotPriceDKK"].mean().reset_index()


##################################### TASK 2 ##################################

#2.1
params = {
    'Pmax': 5,
    'n_c': 0.99,
    'n_d': 0.99,
    'Cmax': 10
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
plt.title("yerly profit from 2019 to 2023")
plt.show()
plt.close()

#2.2

plt.figure()
plt.bar(daily_avg_prices["HourDK"],daylyprofit)
plt.bar(daily_avg_prices["HourDK"],daily_avg_prices["SpotPriceDKK"])
plt.xlabel("year")
plt.ylabel("profit in DKK")
plt.legend(["Profit", "Price"])
plt.title("yerly profit from 2019 to 2023")
plt.show()
plt.close()





#2.3 

params = {
    'Pmax': 5,
    'n_c': 0.95,
    'n_d': 0.95,
    'Cmax': 10
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
    'Pmax': 5,
    'n_c': 0.90,
    'n_d': 0.90,
    'Cmax': 10
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




"""
#2.3 ikke færdig

parms95 = {
    'Pmax': 5,
    'n_c': 0.95,
    'n_d': 0.95,
    'Cmax': 10
}

params['C_0'] = 0.1 * params['Cmax']
params['C_n'] = 0.5 * params['Cmax']


yearprf95 = []


#for day in range(len(df_prices["HourDK"])):
daylyprofit95 = []



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
            daylyprofit95.append(profitOpt)
    profsum95 = sum(prof)
    print("Profit for " + str(year) + ": " + str(profsum95) + " DKK.")
    yearprf95.append(profsum95)
    
    
daylyprofit90 = []
yearprf90 = []


parms90 = {
    'Pmax': 5,
    'n_c': 0.90,
    'n_d': 0.90,
    'Cmax': 10
}

params['C_0'] = 0.1 * params['Cmax']
params['C_n'] = 0.5 * params['Cmax']


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
            daylyprofit90.append(profitOpt)
    profsum90 = sum(prof)
    print("Profit for " + str(year) + ": " + str(profsum90) + " DKK.")
    yearprf90.append(profsum90)


plt.figure()
plt.bar(Yearly["Year"],yearprf)
plt.bar(Yearly["Year"],yearprf95)
plt.bar(Yearly["Year"],yearprf90)
plt.xlabel("year")
plt.ylabel("profit in DKK")
plt.title("yerly profit from 2019 to 2023")

"""
"""
prof2 = []

params = {
    'Pmax': 5,
    'n_c': 0.95,
    'n_d': 0.95,
    'Cmax': 10
}

params['C_0'] = 0.1 * params['Cmax']
params['C_n'] = 0.5 * params['Cmax']

for year in range(2019, 2024):
    t_s = pd.Timestamp(dt.datetime(year, 1, 1, 0, 0, 0))
    t_e = pd.Timestamp(dt.datetime(year, 12, 31, 23, 0, 0))
    p = df_prices.loc[(df_prices["HourDK"]>=t_s) & (df_prices["HourDK"]<=t_e),"Sell"].values
    profitOpt, p_cOpt, p_dOpt, XOpt, daylprof = Optimizer(params, p)
    print("Profit for " + str(year) + ": " + str(profitOpt) + " DKK.")
    #print(daylprof)
    prof2.append(profitOpt)


prof3 = []

params = {
    'Pmax': 5,
    'n_c': 0.90,
    'n_d': 0.90,
    'Cmax': 10
}

params['C_0'] = 0.1 * params['Cmax']
params['C_n'] = 0.5 * params['Cmax']

for year in range(2019, 2024):
    t_s = pd.Timestamp(dt.datetime(year, 1, 1, 0, 0, 0))
    t_e = pd.Timestamp(dt.datetime(year, 12, 31, 23, 0, 0))
    p = df_prices.loc[(df_prices["HourDK"]>=t_s) & (df_prices["HourDK"]<=t_e),"Sell"].values
    profitOpt, p_cOpt, p_dOpt, XOpt, daylprof = Optimizer(params, p)
    print("Profit for " + str(year) + ": " + str(profitOpt) + " DKK.")
    #print(daylprof)
    prof3.append(profitOpt)
    

plt.figure()
plt.bar(Yearly["Year"], prof, label="Efficientcy 0.99")
plt.bar(Yearly["Year"], prof2, label="Efficientcy 0.95")
plt.bar(Yearly["Year"], prof3, label="Efficientcy 0.90")
plt.xlabel("Year")
plt.ylabel("Profit in DKK")
plt.title("Yearly Profit from 2019 to 2023")
plt.legend()
plt.show()


"""


""""
# Define the start and end time to filter your price data
t_s2019 = pd.Timestamp(dt.datetime(2019, 1, 1, 0, 0, 0))
t_e2019 = pd.Timestamp(dt.datetime(2019, 12, 31, 23, 0, 0))


p2019 = df_prices.loc[(df_prices["HourDK"]>=t_s2019) & (df_prices["HourDK"]<=t_e2019),"Sell"].values

profitOpt2019, p_cOpt, p_dOpt, XOpt = Optimizer(params, p2019)

print("The profit is equal to:", profitOpt2019)

t_s2020 = pd.Timestamp(dt.datetime(2020, 1, 1, 0, 0, 0))
t_e2020 = pd.Timestamp(dt.datetime(2020, 12, 31, 23, 0, 0))

p2020 = df_prices.loc[(df_prices["HourDK"]>=t_s2020) & (df_prices["HourDK"]<=t_e2020),"Sell"].values

profitOpt2020, p_cOpt, p_dOpt, XOpt = Optimizer(params, p2020)

print("The profit is equal to:", profitOpt2020)

t_s2021 = pd.Timestamp(dt.datetime(2021, 1, 1, 0, 0, 0))
t_e2021 = pd.Timestamp(dt.datetime(2021, 12, 31, 23, 0, 0))

p2021 = df_prices.loc[(df_prices["HourDK"]>=t_s2021) & (df_prices["HourDK"]<=t_e2021),"Sell"].values

profitOpt2021, p_cOpt, p_dOpt, XOpt = Optimizer(params, p2021)

print("The profit is equal to:", profitOpt2021)

t_s2022 = pd.Timestamp(dt.datetime(2022, 1, 1, 0, 0, 0))
t_e2022 = pd.Timestamp(dt.datetime(2022, 12, 31, 23, 0, 0))

p2022 = df_prices.loc[(df_prices["HourDK"]>=t_s2022) & (df_prices["HourDK"]<=t_e2022),"Sell"].values

profitOpt2022, p_cOpt, p_dOpt, XOpt = Optimizer(params, p2022)

print("The profit is equal to:", profitOpt2022)

t_s2023 = pd.Timestamp(dt.datetime(2023, 1, 1, 0, 0, 0))
t_e2023 = pd.Timestamp(dt.datetime(2023, 12, 31, 23, 0, 0))

p2023 = df_prices.loc[(df_prices["HourDK"]>=t_s2023) & (df_prices["HourDK"]<=t_e2023),"Sell"].values

profitOpt2023, p_cOpt, p_dOpt, XOpt = Optimizer(params, p2023)

print("The profit is equal to:", profitOpt2023)
                       
profit = [profitOpt2019, profitOpt2020, profitOpt2021, profitOpt2022, profitOpt2023]


plt.figure()
plt.bar(Yearly["Year"],profit)
plt.xlabel("year")
plt.ylabel("profit in DKK")
plt.title("yerly profit from 2019 to 2023")
"""






"""""
#ryk rundt på det her
df_pro["Month"] = df_pro["HourDK"].dt.month
df_pro["Year"] = df_pro["HourDK"].dt.year
df_pro["DayOfMonth"] = df_pro["HourDK"].dt.day
df_prices["Month"] = df_prices["HourDK"].dt.month
df_prices["Year"] = df_prices["HourDK"].dt.year
df_prices["Day"] = df_prices["HourDK"].dt.day
df_pro["Buy"] = df_prices["Buy"]
df_pro["Sell"] = df_prices["Sell"]


Net = Netting(df_pro, df_prices)
print("The yearly netting results are: \n", Net[["Year","Profit"]])
"""

