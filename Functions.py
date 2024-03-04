
import pandas as pd

def PricesDK(df_prices):
    
    # Set the Sell price equal to the spot price
    df_prices["Sell"] = df_prices["SpotPriceDKK"]
    
    # Define the fixed Tax and TSO columns
    # multiply by 1000 to get the price in DKK/MWh
    df_prices["Tax"] = 1000*0.8
    df_prices["TSO"] = 1000*0.1
    
    

    ### Add the DSO tariffs ###


    #dso shold be devided by 100 and multiplied by 1000 to get the price in DKK/MWh

    # winter low
    df_prices.loc[(df_prices["HourDK"].dt.month.isin([10,11,12,1,2,3])) 
     & (df_prices["HourDK"].dt.hour.isin([0,1,2,3,4,5,6])), "DSO"] = 15*10


    # winter high
    df_prices.loc[(df_prices["HourDK"].dt.month.isin([10,11,12,1,2,3])) 
     & (df_prices["HourDK"].dt.hour.isin([7,8,9,10,11,12,13,14,15,16,17,22,23])), "DSO"] = 45*10
    
    # winter peak
    df_prices.loc[(df_prices["HourDK"].dt.month.isin([10,11,12,1,2,3])) 
     & (df_prices["HourDK"].dt.hour.isin([18,19,20,21])), "DSO"] = 135*10
    

    #summer low
    df_prices.loc[(df_prices["HourDK"].dt.month.isin([4,5,6,7,8,9])) 
     & (df_prices["HourDK"].dt.hour.isin([0,1,2,3,4,5,6])), "DSO"] = 15*10
    
    # summer high
    df_prices.loc[(df_prices["HourDK"].dt.month.isin([4,5,6,7,8,9])) 
     & (df_prices["HourDK"].dt.hour.isin([7,8,9,10,11,12,13,14,15,16,17,22,23])), "DSO"] = 23*10

    # summer peak
    df_prices.loc[(df_prices["HourDK"].dt.month.isin([4,5,6,7,8,9])) 
     & (df_prices["HourDK"].dt.hour.isin([18,19,20,21])), "DSO"] = 60*10

    
    
    
    
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
    
    """
    # Convert prices to DKK/mwh - tjæk om det er rigtigt
    df_prices['SpotPriceDKK'] = df_prices['SpotPriceDKK']
    """
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




def ProsumerOptimizer(params, l_b, l_s, p_PV, p_L):

    import cvxpy as cp    

    n = len(l_b)
    p_c = cp.Variable(n)
    p_d = cp.Variable(n)
    p_b = cp.Variable(n)
    p_s = cp.Variable(n)
    X   = cp.Variable(n)
    cost = cp.sum(p_b@l_b - p_s@l_s)
    
    constraints = [p_c >= 0, 
                   p_d >= 0, 
                   p_c <= params['Pmax'], 
                   p_d <= params['Pmax'],
                   p_s >= 0,
                   p_b >= 0]
    constraints += [X >= 0, X <= params['Cmax']]
    constraints += [X[0]== params['C_0'] + p_c[0]*params['n_c'] - p_d[0]/params['n_d']]
    constraints += [p_PV + p_b + p_d == p_L + p_s + p_c]
    
    constraints += [X[1:] == X[:-1] + p_c[1:]*params['n_c'] - p_d[1:]/params['n_d']]
    
    constraints += [X[n-1]>=params['C_n']]
    
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.ECOS)
    
    return cost.value, p_c.value, p_d.value, p_b.value, p_s.value, X.value


def Netting(df_pro, df_prices):
    
    import pandas as pd
    
    net_day =[]

   

    # Add buy/sell prices
    df_pro["Buy"] = df_prices["Buy"]
    df_pro["Sell"] = df_prices["Sell"]

    df_pro["Export"] = (df_pro["PV"] - df_pro["Load"]).where(df_pro["PV"] > df_pro["Load"], other=0)
    df_pro["Import"] = (df_pro["Load"] - df_pro["PV"]).where(df_pro["Load"] > df_pro["PV"], other=0)
    df_pro['Profit'] = df_pro["Export"]*df_pro["Sell"] - df_pro["Import"]*df_pro["Buy"]

    net_day.append(df_pro["Profit"].sum()) 
    Net = df_pro.groupby('Year').agg({'Profit': 'sum'}).reset_index()
        
   
    
    return Net


