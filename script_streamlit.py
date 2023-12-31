import yfinance as yf
import requests
import pandas as pd
import numpy as np
from functools import reduce
from datetime import timedelta
import streamlit as st
import plotly.graph_objects as go
from bs4 import BeautifulSoup 

run=False

with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:
        start_period = st.date_input("Start Date")
    with col2:
        end_period = st.date_input("End Date")  

    # no of days to calculate performance
    n_daysmeasure_perf = st.number_input('Number of days to measure the performance for stock selection required for the sample strategy',value=100,step=1)

    # top stock we want
    top = st.number_input('Number of top stocks to be selected for sample strategy',value=10,step=1)

    initial_investment = st.number_input('Initial Equity',value=100000,step=1)

    if st.button('Result'):
        
        run = True


def top_stocks(n=50):
    url = "https://en.wikipedia.org/wiki/NIFTY_50"
    soup = BeautifulSoup(requests.get(url).content,'lxml')
    x=soup.find("table",attrs={"id":"constituents"}).find("tbody").find_all("tr")[1:]

    return [i.findAll("td")[1].text for i in x][:n]


def stock_history(stock,start_period=None,end_period=None,NS=True):
    temp_df = yf.Ticker(f"{stock}.NS") if NS else yf.Ticker(stock)
    temp_df = temp_df.history(start=start_period,end=end_period)[["Open","Close"]]
    temp_df.columns = map(lambda x: x+"_"+stock,temp_df.columns)
    return temp_df


def calc_equity(temp_df,initial_investment,top):
    
    stock = temp_df.columns[0].split("_")[-1]
    temp_df[f"Daily Value_{stock}"]=temp_df.iloc[:,1]*(initial_investment/top/temp_df.iloc[0,0])
    temp_df.reset_index(inplace=True)
    temp_df['Date']=pd.to_datetime(temp_df['Date'])
    temp_df['Date']=temp_df['Date'].dt.date
    return temp_df

def give_valid(date,df):
    while len(df[df["Date"]==date])!=1:
        date = date-timedelta(1)
    return date

def top_best_per_stock(df,top,n_daysmeasure_perf):
    last_date = give_valid(df["Date"].max()-timedelta(1),df)
    start_date = give_valid(last_date-timedelta(n_daysmeasure_perf),df)
    
    temp_df = df[df["Date"].isin([last_date,start_date])]
    temp_df=temp_df[[i for i in temp_df.columns if "Close" in i]]
    
    temp_df.iloc[0]=temp_df.iloc[0]-1
    temp_df=(temp_df.iloc[1]/(temp_df.iloc[0]-1)).sort_values(ascending=False)
    
    return [i.replace("Close_","") for i in temp_df[:top].index]

def performance(df,t):
    """_summary_

    Args:
        df (pd.Series): column
        t (float/int): year

    Returns:
        tupple: CAGR (%), Volatility (%), Sharpe Ratio
    """
    d_r = ((df/df.shift(1))-1).dropna()     # daily return
    std_d_r = np.std(d_r)
    
    return ((df.iloc[-1]/df.iloc[0])**(1/t)-1)*100, (std_d_r**(1/252))*100, (np.mean(d_r)/std_d_r)**(1/252)




if run:

    df = reduce(lambda x,y : pd.merge(x,calc_equity(stock_history(y,
                                                                start_period,
                                                                end_period),
                                                    initial_investment,
                                                    50),
                                    on="Date",
                                    how="outer"),
                top_stocks(),
                pd.DataFrame(columns=["Date"]))

    df['Equity Curve']=df[[col for col in df.columns if "Daily Value" in col]].sum(axis=1)

    # removing daily values as we dont need them
    df.drop([col for col in df.columns if "Daily Value" in col],axis=1,inplace=True)



    top_stocks_names = top_best_per_stock(df,top,n_daysmeasure_perf)

    new_df=reduce(lambda x,y : pd.merge(x,calc_equity(df[[col for col in df.columns if y in col]].set_index([df["Date"]]),
                                                initial_investment,
                                                top),
                                    on="Date",
                                    how="outer"),
                top_stocks_names,
                pd.DataFrame(columns=["Date"]))

    new_df['Portfolio Equity Curve']=new_df[[col for col in new_df.columns if "Daily" in col]].sum(axis=1)

    df=pd.merge(df[["Date","Equity Curve"]],new_df[["Date","Portfolio Equity Curve"]],on="Date")

    new_df=calc_equity(stock_history("^NSEI",
                                start_period,
                                end_period,NS=False),
                initial_investment,top=1)

    df=pd.merge(df[["Date","Equity Curve","Portfolio Equity Curve"]],new_df[["Date","Daily Value_^NSEI"]],on="Date").rename(columns={"Daily Value_^NSEI":"Nifty Index Equity Curve"})


    # Calculating the number of years
    t=(pd.to_datetime(end_period) - pd.to_datetime(start_period)) / timedelta(365)

    final_df=pd.DataFrame({"Equally Alloc Buy Hold":performance(df['Equity Curve'],t),
                            "Nifty":performance(df['Nifty Index Equity Curve'],t),
                            "Performance_Strat":performance(df['Portfolio Equity Curve'],t)},
                        index=["CAGR (%)", "Volatility (%)", "Sharpe Ratio"])
    
    st.write(final_df.T)
    
    fig=go.Figure()

    temp_store = []

    fig.add_scatter(x=df['Date'],y=df["Equity Curve"],name="Equally Alloc Buy Hold",marker=dict(color="#ff0000"))
    fig.add_scatter(x=df['Date'],y=df["Nifty Index Equity Curve"],name="Nifty",marker=dict(color="#00f7ff"))
    fig.add_scatter(x=df['Date'],y=df["Portfolio Equity Curve"],name="Performance_Strat",marker=dict(color="#11ff00"))
    fig.update_layout(hovermode="x unified")
    
    st.plotly_chart(fig,use_container_width=True)
    
    
    st.write("Top Stocks Selected:")
    st.write(top_stocks_names)