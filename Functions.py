
import pandas as pd

def PricesDK(df_prices):
    
    # Set the Sell price equal to the spot price
    df_prices["Sell"] = df_prices["SpotPriceDKK"]
    
    # Define the fixed Tax and TSO columns
    df_prices["Tax"] = 0.8
    df_prices["TSO"] = 0.1
    
    

    ### Add the DSO tariffs ###


    # winter low
    df_prices.loc[(df_prices["HourDK"].dt.month.isin([10,11,12,1,2,3])) 
     & (df_prices["HourDK"].dt.hour.isin([0,1,2,3,4,5,6])), "DSO"] = 15


    # winter high
    df_prices.loc[(df_prices["HourDK"].dt.month.isin([10,11,12,1,2,3])) 
     & (df_prices["HourDK"].dt.hour.isin([7,8,9,10,11,12,13,14,15,16,17,22,23])), "DSO"] = 45
    
    # winter peak
    df_prices.loc[(df_prices["HourDK"].dt.month.isin([10,11,12,1,2,3])) 
     & (df_prices["HourDK"].dt.hour.isin([18,19,20,21])), "DSO"] = 135
    

    #summer low
    df_prices.loc[(df_prices["HourDK"].dt.month.isin([4,5,6,7,8,9])) 
     & (df_prices["HourDK"].dt.hour.isin([0,1,2,3,4,5,6])), "DSO"] = 15
    
    # summer high
    df_prices.loc[(df_prices["HourDK"].dt.month.isin([4,5,6,7,8,9])) 
     & (df_prices["HourDK"].dt.hour.isin([7,8,9,10,11,12,13,14,15,16,17,22,23])), "DSO"] = 23

    # summer peak
    df_prices.loc[(df_prices["HourDK"].dt.month.isin([4,5,6,7,8,9])) 
     & (df_prices["HourDK"].dt.hour.isin([18,19,20,21])), "DSO"] = 60

    
    
    
    
    # Calculate VAT
    df_prices["VAT"] = 0.25*(df_prices["SpotPriceDKK"]+df_prices["TSO"]+df_prices["DSO"]+df_prices["Tax"])
    
    # Calculate Buy price
    df_prices["Buy"] = df_prices["VAT"]+df_prices["SpotPriceDKK"]+df_prices["TSO"]+df_prices["DSO"]+df_prices["Tax"]
    
    return df_prices

def LoadData():
    
    import os
    import pandas as pd
    
    ### Load electricity prices ###
    price_path = os.path.join(os.getcwd(),'ElspotpricesEA.csv')
    df_prices = pd.read_csv(price_path)
    
    # Convert to datetime
    df_prices["HourDK"] = pd.to_datetime(df_prices["HourDK"])
    
    # Convert prices to DKK/mwh - tjæk om det er rigtigt
    df_prices['SpotPriceDKK'] = df_prices['SpotPriceDKK']/(1000^2)
    
    # Filter only DK2 prices
    df_prices = df_prices.loc[df_prices['PriceArea']=="DK2"]
    
    # Keep only the local time and price columns
    df_prices = df_prices[['HourDK','SpotPriceDKK',"HourUTC"]]
    
    # Keep only 2022 and 2023
    #df_prices = df_prices.loc[df_prices["HourDK"].dt.year.isin([2018,2019,2020,2021,2022,2023])]
    
    # Reset the index
    df_prices = df_prices.reset_index(drop=True)
    
    ###  Load prosumer data ###
    file_P = os.path.join(os.getcwd(),'ProsumerHourly.csv')
    df_pro = pd.read_csv(file_P)
    df_pro["TimeDK"] = pd.to_datetime(df_pro["TimeDK"])
    df_pro = df_pro.reset_index(drop=True)
    df_pro.rename(columns={'Consumption': 'Load'}, inplace=True)
    df_pro.rename(columns={'TimeDK': 'HourDK'}, inplace=True)

    return df_prices, df_pro



def Optimizer(params, p):

    import cvxpy as cp

    n = len(p)
    p_c = cp.Variable(n)
    p_d = cp.Variable(n)
    X   = cp.Variable(n)
    profit = cp.sum(p_d@p - p_c@p)
    daylyprof = []
    for i in range(n):
        daylyprof.append(p_d@p - p_c@p)


    constraints = [p_c >= 0, 
                   p_d >= 0, 
                   p_c <= params['Pmax'], 
                   p_d <= params['Pmax']]
    constraints += [X >= 0, X <= params['Cmax']]
    constraints += [X[0]==params['C_0'] + p_c[0]*params['n_c'] - p_d[0]/params['n_d']]
    
    constraints += [X[1:] == X[:-1] + p_c[1:]*params['n_c'] - p_d[1:]/params['n_d']]
    
    constraints += [X[n-1]>=params['C_n']]
    
    problem = cp.Problem(cp.Maximize(profit), constraints)
    problem.solve(solver=cp.ECOS)
    
    return profit.value, p_c.value, p_d.value, X.value, daylyprof
     
     
""""
#ikke god 
def Netting(df_pro, df_prices):



    # Calculate yearly price statistics
        df_prices_mean = df_prices.groupby('Year').agg({'Buy': 'mean', 'Sell': 'mean'}).reset_index()

        # Calculate yearly statistics
        df_sum = df_pro.groupby('Year').agg({'PV': 'sum', 'Load': 'sum'}).reset_index()

        # Calculate yearly Imports/Exports
        df_sum["Export"] = (df_sum["PV"] - df_sum["Load"]).where(df_sum["PV"] > df_sum["Load"], other=0)
        df_sum["Import"] = (df_sum["Load"] - df_sum["PV"]).where(df_sum["Load"] > df_sum["PV"], other=0)

        Net = pd.merge(df_prices_mean, df_sum, on='Year')
        Net['Profit'] = Net["Export"]*Net["Sell"] - Net["Import"]*Net["Buy"]

        return Net
"""