# %%
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns 
st.set_page_config(layout="wide")
st.title("Monte Carlo Option Pricing App")
st.markdown("Supports European, Asian , and American Options ")

option_type = st.sidebar.selectbox("Option Type", ["Barrier","American"])


if option_type == "American":
 # Sidebar inputs
 st.sidebar.header("Input Parameters")
 symbol = st.sidebar.text_input("Stock Symbol", "INFY.NS")
 start_str = st.sidebar.text_input("Start Date (YYYY-MM-DD)", "2025-01-01")
 end_str = st.sidebar.text_input("End Date (YYYY-MM-DD)", "2025-06-30")

 start_date = pd.to_datetime(start_str)
 end_date = pd.to_datetime(end_str)

 #Downloading Infosys data from Jan 2025 to June 2025
 data = yf.download("INFY.NS", start=start_date, end=end_date, auto_adjust=False)

 #Calculate daily returns and volatility
 data['Return'] = data['Close'].pct_change()
 data.dropna(inplace=True)
 daily_vol = data['Return'].std()
 vol = daily_vol * np.sqrt(252)

  #Set option parameters
 lcallt_price = data['Close'].iloc[-1].item()
 S = st.sidebar.number_input("Stock Price (S0)", value=float(lcallt_price), step=1.0,key="stock price")
 K = st.sidebar.number_input("Strike Price (K)", value=float(S * 1.1), step=1.0,key="strike price")
 T = st.sidebar.number_input("Time to Maturity (T in years)", value=0.5,key="time")
 r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05,key="risk")
 N = st.sidebar.number_input("Time Steps (N)", value=250,key="time steps")
 M = st.sidebar.number_input("Simulations (M)", value=10000,key="simula")

 dt = T / N
 st.subheader("🔔 American Option Pricing")
 @st.cache_data
 # binomial tree method for american options 
 def american_option_binomial(S, K, T, r, vol, N, option_type='put'):
    dt = T / N
    u = np.exp(vol * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Build price tree
    price_tree = np.zeros((N+1, N+1))
    for i in range(N+1):
        for j in range(i+1):
            price_tree[j, i] = S * (u ** (i - j)) * (d ** j)

    # Option value at maturity
    option = np.zeros((N+1, N+1))
    if option_type == 'put':
        option[:, N] = np.maximum(K - price_tree[:, N], 0)
    else:
        option[:, N] = np.maximum(price_tree[:, N] - K, 0)

    #working backwards
    for i in range(N-1, -1, -1):
        for j in range(i+1):
            cont_val = np.exp(-r * dt) * (p * option[j, i+1] + (1 - p) * option[j+1, i+1])
            if option_type == 'put':
                payoffs = K - price_tree[j, i]
            else:
                payoffs = price_tree[j, i] - K
            option[j, i] = max(cont_val, payoffs)

    return option[0, 0]

 p = american_option_binomial(S, K, T, r, vol, N, option_type='put')
 c = american_option_binomial(S, K, T, r, vol, N, option_type='call')

 @st.cache_data
 def generate_stock_paths(S,r,vol, T, M, N,seed = None):
    if seed is not None:
        np.random.seed(seed)
    dt = T / N
    S_paths = np.zeros((M, N+1))
    
    S_paths[:, 0] = S
    for t in range(1, N+1):
        Z = np.random.standard_normal(M)
        S_paths[:, t] = S_paths[:, t-1] * np.exp((r - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * Z)
        
    return S_paths

 S_paths = generate_stock_paths(S, r, vol, T, M, N, seed=None)

 @st.cache_data
 #lecallt square monte carlo method for american options 
 def lsmc_option(S_paths, K, r, dt, option_type='put'):
        M, N = S_paths.shape[0], S_paths.shape[1] - 1
        if option_type == 'put':
            payoff = np.maximum(K - S_paths[:, -1], 0)
        else:
            payoff = np.maximum(S_paths[:, -1] - K, 0)

        ccallhflow = payoff.copy()

        for t in range(N - 1, 0, -1):
            if option_type == 'put':
                itm = np.where(K - S_paths[:, t] > 0)[0]
                intrinsic = K - S_paths[itm, t]
            else:
                itm = np.where(S_paths[:, t] - K > 0)[0]
                intrinsic = S_paths[itm, t] - K

            if len(itm) == 0:
                continue

            X = S_paths[itm, t]
            Y = ccallhflow[itm] * np.exp(-r * dt)

            # Polynomial bcallis functions: [1, X, X²]
            A = np.vstack([np.ones_like(X), X, X**2]).T
            coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]
            continuation = coeffs[0] + coeffs[1] * X + coeffs[2] * X**2

            exercise = intrinsic > continuation
            exercise_indices = itm[exercise]
            ccallhflow[exercise_indices] = intrinsic[exercise]

        option_price = np.mean(ccallhflow * np.exp(-r * T))
        return option_price

 def plot_simulated_paths(S, vol, r, N, M, T, seed=20):
        np.random.seed(seed)
        dt = T / N
        drift = (r - 0.5 * vol ** 2) * dt
        random_term = vol * np.sqrt(dt)
        lnS = np.log(S_paths)

        Z = np.random.normal(size=(N, M))
        delta_lnSt = drift + random_term * Z
        lnS_initial = lnS[:, 0]  
        lnS_initial = lnS[:, 0].reshape(1, M)  
        lnSt = np.concatenate((np.full((1, M), lnS_initial), delta_lnSt), axis=0)
        lnst = np.cumsum(lnSt, axis=0)

        St = np.exp(lnst)

        t = np.arange( N + 1)
    
       # Plot first 10 paths
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axhline(np.mean(S), color='red', linestyle='--', label=f"Stock ₹{np.mean(S):.2f}")
        ax.axhline(K, color='green', linestyle='--', label=f"Strike ₹{K:.2f}")
        for i in range(min(200, M)):
           plt.plot(t, St[:, i], lw=1)
        ax.set_title("Simulated GBM Paths for Infosys")
        ax.set_xlabel("Time (Days)")
        ax.set_ylabel("Stock Price")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        return Z 

  # Step 6: Run LSMC for both Put and Call
 lsmc_put_price = lsmc_option(S_paths, K, r, dt, option_type='put')
 lsmc_call_price = lsmc_option(S_paths, K, r, dt, option_type='call')

 box2 = st.container(border=True)
 box3 = st.container(border=True)
 tab1, tab2, tab3 = st.tabs(['Option prices','Greeks','Price convergence graphs'])
 
 with tab1:
   binomial_put_price = american_option_binomial(S, K, T, r, vol, N, option_type='put')
   binomial_call_price= american_option_binomial(S, K, T, r, vol, N, option_type='call')
   lsmc_put_price = lsmc_option(S_paths, K, r, dt, option_type='put')
   lsmc_call_price = lsmc_option(S_paths, K, r, dt, option_type='call')
   box1 = st.container(border=True)
   with box1: 
        col1, col2 = box1.columns(2)
        col1.metric("American Call option (Binomial)", f"₹{binomial_call_price:.2f}")
        col1.metric("American put option (Binomial)", f"₹{binomial_put_price:.2f}")
        col2.metric("LSMC Call option", f"₹{lsmc_call_price:.2f}")
        col2.metric("LSMC Put option", f"₹{lsmc_put_price:.2f}")

   box4 = st.container(border=True)
   with box4:
        # Plot 10 simulated paths
        box4.subheader("Simulated Stock Price Paths")
        Z = plot_simulated_paths(S, vol, r, N, M=10000, T=T)

 n_steps = 10000
 step_size = 100
 n_simulations_list = np.arange(100, 10000, step_size)
 dt = T / N
 lsmc_put = []
 lsmc_call= []
 for n in n_simulations_list:
        S_paths = np.zeros((n, N+1))
        S_paths[:, 0] = S
        for t in range(1, N+1):
            Z = np.random.standard_normal(n)  # n samples for n simulations
            S_paths[:, t] = S_paths[:, t-1] * np.exp((r - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * Z)
        put_price = lsmc_option(S_paths, K, r, dt, option_type='put')
        lsmc_put.append(put_price)
        call_price = lsmc_option(S_paths, K, r, dt, option_type='call')
        lsmc_call.append(call_price)
 
 @st.cache_data
 def american_binomial_greeks(S,K,T,r,vol):

        Delta_Put = american_option_binomial(S+0.5, K, T, r, vol, N, option_type='put') - american_option_binomial(S-0.5, K, T, r, vol, N, option_type='put')
        Delta_Call = american_option_binomial(S+0.5, K, T, r, vol, N, option_type='call') - american_option_binomial(S-0.5, K, T, r, vol, N, option_type='call')

        Gamma_Put = 4*(american_option_binomial(S+0.5, K, T, r, vol, N, option_type='put') + american_option_binomial(S-0.5, K, T, r, vol, N, option_type='put') - (2*american_option_binomial(S, K, T, r, vol, N, option_type='put')))
        Gamma_Call = 4*(american_option_binomial(S+0.5, K, T, r, vol, N, option_type='call') + american_option_binomial(S-0.5, K, T, r, vol, N, option_type='call') - (2*american_option_binomial(S, K, T, r, vol, N, option_type='call')))


        dvol = 0.001
        vega_put = ((american_option_binomial(S, K, T, r, vol + dvol, N, option_type='put'))-(american_option_binomial(S, K, T, r, vol-dvol, N, option_type='put')))/(2*dvol)
        vega_call = ((american_option_binomial(S, K, T, r, vol + dvol, N, option_type='call'))-(american_option_binomial(S, K, T, r, vol-dvol, N, option_type='call')))/(2*dvol)

        delta_T = 1 / 252  # 1 trading day
        T_theta = T - delta_T

        theta_put = american_option_binomial(S, K, T_theta, r, vol, N, option_type='put')
        theta_call = american_option_binomial(S, K, T_theta, r, vol, N, option_type='call')
        dr = 0.001

        put_r_plus = american_option_binomial(S, K, T, r + dr, vol, N, option_type='put')
        call_r_plus = american_option_binomial(S, K, T, r + dr, vol, N, option_type='call')

        # r - dr

        put_r_minus =  american_option_binomial(S, K, T, r - dr, vol, N, option_type='put')
        call_r_minus =  american_option_binomial(S, K, T, r - dr, vol, N, option_type='call')

        rho_put = (put_r_plus - put_r_minus) / (2 * dr)
        rho_call = (call_r_plus - call_r_minus) / (2 * dr)

        return  Delta_Call, Delta_Put , Gamma_Call, Gamma_Put, vega_call, vega_put, theta_call, theta_put, rho_call, rho_put 



 @st.cache_data
 def american_lsmc_greeks(S,K,r,vol,T,N,M):
        h = 1.0
        dt = T / N
        seed = 42
        base_paths = generate_stock_paths(S, r, vol, T, M, N, seed = 42)
        price_call = lsmc_option(base_paths, K, r, dt, 'call')
        price_put = lsmc_option(base_paths, K, r, dt, 'put')

        paths_plus = generate_stock_paths(S + h, r, vol, T, M, N,seed = 42)
        paths_minus = generate_stock_paths(S - h, r, vol, T, M, N,seed = 42)
        delta_call = (lsmc_option(paths_plus, K, r, dt, 'call') - lsmc_option(paths_minus, K, r, dt, 'call')) / (2 * h)
        gamma_call = (lsmc_option(paths_plus, K, r, dt, 'call') - 2 * price_call + lsmc_option(paths_minus, K, r, dt, 'call')) / (h ** 2)
        delta_put = (lsmc_option(paths_plus, K, r, dt, 'put') - lsmc_option(paths_minus, K, r, dt, 'put')) / (2 * h)
        gamma_put = (lsmc_option(paths_plus, K, r, dt, 'put') - 2 * price_put + lsmc_option(paths_minus, K, r, dt, 'put')) / (h ** 2)

        vega_call = (lsmc_option(generate_stock_paths(S, r, vol + 0.001, T, M, N,seed = 42), K, r, dt, 'call') -
                    lsmc_option(generate_stock_paths(S, r, vol - 0.001, T, M, N,seed = 42), K, r, dt, 'call')) / (2 * 0.001)
        vega_put = (lsmc_option(generate_stock_paths(S, r, vol + 0.001, T, M, N,seed = 42), K, r, dt, 'put') -
                    lsmc_option(generate_stock_paths(S, r, vol - 0.001, T, M, N,seed = 42), K, r, dt, 'put')) / (2 * 0.001)

        rho_call = (lsmc_option(generate_stock_paths(S, r + 0.001, vol, T, M, N,seed = 42), K, r + 0.001, dt, 'call') -
                    lsmc_option(generate_stock_paths(S, r - 0.001, vol, T, M, N,seed = 42), K, r - 0.001, dt, 'call')) / (2 * 0.001)
        rho_put = (lsmc_option(generate_stock_paths(S, r + 0.001, vol, T, M, N,seed = 42), K, r + 0.001, dt, 'put') -
                lsmc_option(generate_stock_paths(S, r - 0.001, vol, T, M, N,seed = 42), K, r - 0.001, dt, 'put')) / (2 * 0.001)

        T_h = 1 / 365
        dt_h = (T - T_h) / N
        theta_call = (lsmc_option(generate_stock_paths(S, r, vol, T - T_h, M, N,seed = 42), K, r, dt_h, 'call') - price_call) / (-T_h)
        theta_put = (lsmc_option(generate_stock_paths(S, r, vol, T - T_h, M, N,seed = 42), K, r, dt_h, 'put') - price_put) / (-T_h)

        return delta_call, delta_put, gamma_call, gamma_put, vega_call, vega_put, theta_call, theta_put, rho_call, rho_put

 greeks = {'delta_call': [],'delta_put': [], 'gamma_call':[], 'gamma_put':[], 'vega_call':[], 'vega_put':[], 'theta_call':[], 'theta_put':[], 'rho_call':[], 'rho_put':[]}
 S_rang = np.linspace(S * 0.8, S * 1.2, 50)

 for S_val in S_rang:
        delta_call, delta_put, gamma_call, gamma_put, vega_call, vega_put, theta_call, theta_put, rho_call, rho_put = american_lsmc_greeks(S_val,K,r,vol,T,N,M)
        greeks['delta_call'].append(delta_call)
        greeks['gamma_call'].append(gamma_call)
        greeks['vega_call'].append(vega_call)
        greeks['theta_call'].append(theta_call)
        greeks['rho_call'].append(rho_call)
        greeks['delta_put'].append(delta_put)
        greeks['gamma_put'].append(gamma_put)
        greeks['vega_put'].append(vega_put)
        greeks['theta_put'].append(theta_put)
        greeks['rho_put'].append(rho_put)

 with tab2:
    box2 = st.container(border=True)
    with box2:
        delta_call_bi,delta_put_bi,gamma_call_bi,gamma_put_bi,vega_call_bi,vega_put_bi,theta_call_bi,theta_put_bi,rho_call_bi,rho_put_bi = american_binomial_greeks(S,K,T,r,vol)
        delta_call,delta_put,gamma_call,gamma_put,vega_call,vega_put,theta_call,theta_put,rho_call,rho_put = american_lsmc_greeks(S,K,r,vol,T,N,M)
        box2.subheader("Greeks for American Options by Binomial Method ")

        ro1,ro2,ro3,ro4,ro5 = box2.columns(5)
        ro1.metric("Delta Call",f"₹{delta_call_bi:.2f}")
        ro1.metric("Delta Put",f"₹{delta_put_bi:.2f}")
        ro2.metric("Gamma Call ", f"₹{abs(gamma_call_bi):.2f}")
        ro2.metric("Gamma put",f"₹{abs(gamma_put_bi):.2f}")
        ro3.metric("Vega Call ",f"₹{vega_call_bi:.2f}")
        ro3.metric("vega put",f"₹{vega_put_bi:.2f}")
        ro4.metric("theta Call ",f"₹{theta_call_bi:.2f}")
        ro4.metric("theta put",f"₹{theta_put_bi:.2f}")
        ro5.metric("Rho Call ",f"₹{rho_call_bi:.2f}")
        ro5.metric("rho put",f"₹{rho_put_bi:.2f}")



    box3 = st.container(border=True)
    with box3:
        box3.subheader("Greeks for American Options by LSMC method")
        col4, col5 , col6 , col7,col8 = box3.columns(5)
        col4.metric("Delta Call ",f"₹{delta_call:.2f}")
        col4.metric("Delta put",f"₹{delta_put:.2f}")
        col5.metric("Gamma Call ", f"₹{abs(gamma_call) :.2f}")
        col5.metric("Gamma put",f"₹{abs(gamma_put):.2f}")
        col6.metric("Vega Call ",f"₹{vega_call:.2f}")
        col6.metric("vega put",f"₹{vega_put:.2f}")
        col7.metric("theta Call ",f"₹{theta_call:.2f}")
        col7.metric("theta put",f"₹{theta_put:.2f}")
        col8.metric("Rho Call ",f"₹{rho_call:.2f}")
        col8.metric("rho put",f"₹{rho_put:.2f}")

   
    col1, col2 = tab2.columns([2, 1])

    with col2:
        st.header("Input Parameters")
        K = st.number_input("Strike Price (K)", value=100.0, key="strike-price")
        T = st.number_input("Time to Maturity (T in years)", value=1.0, key="tiime")
        r = st.number_input("Risk-Free Rate (r)", value=0.05, key="riksk")
        vol = st.number_input("Volatility (vol)", value=0.2, key="volatiglity")
        M = st.number_input("Number of Simulations", value=10000, step=1000, key="sigmula")
        S_min = st.number_input("Min Stock Price", value=50.0, key="min")
        S_max = st.number_input("Max Stock Price", value=150.0, key="max")
        S_step = st.number_input("Step Size", value=5.0, key="stegp")

 # Compute Greeks over range of stock prices
    S_range = np.linspace(80, 120, len(greeks['delta_call']))

    with col1:
        st.header("Greeks vs Stock Price")
        fig, axis = plt.subplots(5, 2, figsize=(14, 10))

        axis[0, 0].plot(S_range, greeks['delta_call'], color='blue')
        axis[0, 0].set_title("Delta Callvs Stock Price")

        axis[0, 1].plot(S_range, greeks['delta_put'], color='blue')
        axis[0, 1].set_title("Delta Put vs Stock Price")

        axis[1,0].plot(S_range, greeks['gamma_call'], color='green')
        axis[1,0].set_title("Gamma Call vs Stock Price")

        axis[1, 1].plot(S_range, greeks['gamma_put'], color='green')
        axis[1, 1].set_title("Gamma Put vs Stock Price")

        axis[2,0].plot(S_range, greeks['vega_call'], color='purple')
        axis[2, 0].set_title("Vega Call vs Stock Price")

        axis[2,1].plot(S_range, greeks['vega_put'], color='purple')
        axis[2,1].set_title("Vega Put vs Stock Price")

        axis[3,0].plot(S_range, greeks['theta_call'], color='red')
        axis[3,0].set_title("Theta Call vs Stock Price")

        axis[3, 1].plot(S_range, greeks['theta_put'], color='red')
        axis[3, 1].set_title("Theta Put vs Stock Price")

        axis[4, 0].plot(S_range, greeks['rho_call'], color='orange')
        axis[4, 0].set_title("Rho Call vs Stock Price")
        
        axis[4,1].plot(S_range, greeks['rho_put'], color='orange')
        axis[4,1].set_title("Rho Put vs Stock Price")

        for ax in axis.flatten():
                ax.set_xlabel("Stock Price")
                ax.set_ylabel("Greek Value")
                ax.grid(True)

        fig.tight_layout()
        st.pyplot(fig)

 with tab3:
    fig, ax = plt.subplots(1,2,figsize=(12,6))
    ax[0].semilogx(n_simulations_list, lsmc_put,alpha = 0.7, label = "Least square Monte Carlo")
    ax[0].axhline(y =p,color= 'red',linestyle='--', alpha = 0.7,label='Binomial Tree Method')
    ax[0].set_xlabel("number of simulations")
    ax[0].set_ylabel("option price")
    ax[0].set_title("price convergence of Put price option")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].semilogx(n_simulations_list, lsmc_call,alpha = 0.7, label = "Least square Monte Carlo")
    ax[1].axhline(y = c,color= 'red',linestyle='--', alpha = 0.7,label='Binomial Tree Method')
    ax[1].set_xlabel("number of simulations")
    ax[1].set_ylabel("option price")
    ax[1].set_title("price convergence of call price option")
    ax[1].legend()
    ax[1].grid(True)

    fig.tight_layout()
    st.pyplot(fig)
     
if option_type == "Barrier":
 
 st.subheader("🔔 Barrier Option Pricing (Up-In / Down-In)")
 st.sidebar.header("📌 Option Parameters")
 ticker_symbol = st.sidebar.text_input("Ticker", value="INFY.NS", key="hah8")
 K = st.sidebar.number_input("Strike Price (K)", value=1550.0, key="hah7")
 barrier_type = st.sidebar.selectbox("Barrier Type", ["up-in", "down-in"], key="hah6")
 option_kind = st.sidebar.selectbox("Option Type", ["call", "put"], key="hah5")
 T = st.sidebar.slider("Time to Maturity (Years)", 0.1, 2.0, 1.0, key="hah3")
 r = st.sidebar.slider("Risk-Free Rate (r)", 0.00, 0.20, 0.06, key="hah4")
 n_steps = st.sidebar.slider("Time Steps", 50, 365, 252, key="hah12")
 n_paths = st.sidebar.slider("Simulations", 1000, 50000, 10000, step=1000, key="hah1")
 
 data = yf.Ticker(ticker_symbol).history(period="6mo")
 if data.empty:
    st.error("Failed to fetch data. Check the ticker symbol.")
    st.stop()

 S = data['Close'].iloc[-1]
 returns = data['Close'].pct_change().dropna()
 sigma = returns.std() * np.sqrt(252)

 barrier = S + 50 if barrier_type == "up-in" else S - 50

# --- Simulate Paths ---
 def simulate_paths(S0, T, r, sigma, n_steps, n_paths):
    dt = T / n_steps
    S_paths = np.zeros((n_paths, n_steps + 1))
    S_paths[:, 0] = S0
    for t in range(1, n_steps + 1):
        Z = np.random.standard_normal(n_paths)
        S_paths[:, t] = S_paths[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return S_paths


 S_paths = simulate_paths(S, T, r, sigma, n_steps, n_paths)
 final = S_paths[:, -1]
 barrier_hit = np.any(S_paths >= barrier, axis=1) if "up" in barrier_type else np.any(S_paths <= barrier, axis=1)
 valid = barrier_hit

 # --- Payoffs ---
 call_payoff = np.maximum(final - K, 0)
 put_payoff = np.maximum(K - final, 0)
 call_price = np.exp(-r * T) * np.mean(call_payoff * valid)
 put_price = np.exp(-r * T) * np.mean(put_payoff * valid)

 tab1 , tab2 , tab3 = st.tabs(['Option Prices','Greeks',"Price convergence graphs"])

 with tab1: 
    # --- Display Option Prices ---
    st.subheader("💰 Option Prices")
    st.markdown(f"**Spot Price (S₀):** {S:.2f}")
    st.markdown(f"**Barrier Level (B):** {barrier:.2f}")
    col1, col2 = st.columns(2)
    col1.metric("Call Option ({}):".format(barrier_type), f"₹ {call_price:.2f}")
    col2.metric("Put Option ({}):".format(barrier_type), f"₹ {put_price:.2f}")

    #-- Payoff Distribution ---
    st.subheader("📊 Payoff Distribution")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.histplot(call_payoff * valid if option_kind == "call" else put_payoff * valid, bins=50, kde=True, ax=ax1, color='skyblue')
    ax1.set_title("Payoff Distribution")
    ax1.set_xlabel("Payoff")
    ax1.set_ylabel("Frequency")
    st.pyplot(fig1)


 # --- Greeks ---
 def compute_greeks(S, sigma, r, is_call):
    price = price_mc(S, sigma, r, is_call)
    price_up = price_mc(S + 1, sigma, r, is_call)
    price_down = price_mc(S - 1, sigma, r, is_call)
    delta = (price_up - price_down) / 2
    gamma = (price_up - 2 * price + price_down)
    vega = (price_mc(S, sigma + 0.01, r, is_call) - price) / 0.01
    theta = (price - price) / (-1/252)
    rho = (price_mc(S, sigma, r + 0.01, is_call) - price) / 0.01
    return price, delta, gamma, vega, theta, rho

 # --- Price Monte Carlo ---
 def price_mc(S, sigma_local, r_local, is_call):
    paths = simulate_paths(S, T, r_local, sigma_local, n_steps, n_paths)
    final = paths[:, -1]
    barrier_hit = np.any(paths >= barrier, axis=1) if "up" in barrier_type else np.any(paths <= barrier, axis=1)
    valid = barrier_hit
    payoff = np.maximum(final - K, 0) if is_call else np.maximum(K - final, 0)
    return np.exp(-r_local * T) * np.mean(payoff * valid)

 with tab2: 
    # --- Display Greeks Separately ---
    st.subheader("📉 Greeks (Call)")
    _, delta, gamma, vega, theta, rho = compute_greeks(S, sigma, r, True)
    colc = st.columns(5)
    colc[0].metric("Delta", f"{delta:.4f}")
    colc[1].metric("Gamma", f"{abs(gamma):.4f}")
    colc[2].metric("Vega", f"{vega:.4f}")
    colc[3].metric("Theta", f"{theta:.4f}")
    colc[4].metric("Rho", f"{rho:.4f}")

    st.subheader("📉 Greeks (Put)")
    _, delta, gamma, vega, theta, rho = compute_greeks(S, sigma, r, False)
    colp = st.columns(5)
    colp[0].metric("Delta", f"{delta:.4f}")
    colp[1].metric("Gamma", f"{abs(gamma):.4f}")
    colp[2].metric("Vega", f"{vega:.4f}")
    colp[3].metric("Theta", f"{theta:.4f}")
    colp[4].metric("Rho", f"{rho:.4f}")

    # --- Greeks vs Volatility ---
    st.subheader("🌀 Greeks vs Volatility")
    vol_range = np.linspace(0.1, 0.6, 10)
    deltas, gammas, vegas, thetas, rhos = [], [], [], [], []

    for vol in vol_range:
        price = price_mc(S, vol, r, option_kind == "call")
        price_up = price_mc(S + 1, vol, r, option_kind == "call")
        price_down = price_mc(S - 1, vol, r, option_kind == "call")
        deltas.append((price_up - price_down) / 2)
        gammas.append((price_up - 2 * price + price_down))
        vegas.append((price_mc(S, vol + 0.01, r, option_kind == "call") - price) / 0.01)
        thetas.append((price - price) / (-1/252))
        rhos.append((price_mc(S, vol, r + 0.01, option_kind == "call") - price) / 0.01)

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(vol_range, deltas, label='Delta')
    ax3.plot(vol_range, gammas, label='Gamma')
    ax3.plot(vol_range, vegas, label='Vega')
    ax3.plot(vol_range, thetas, label='Theta')
    ax3.plot(vol_range, rhos, label='Rho')
    ax3.set_title("Greeks vs Volatility")
    ax3.set_xlabel("Volatility (σ)")
    ax3.set_ylabel("Value")
    ax3.legend()
    st.pyplot(fig3)

 with tab3:
    K = st.number_input("Strike Price (K)", value=1550.0)
    barrier_type = st.selectbox("Barrier Type", ["up-in", "down-in"], key ="jiji")
    option_kind = st.selectbox("Option Type", ["call", "put"], key="option")
    T = st.slider("Time to Maturity (Years)", 0.1, 2.0, 1.0, key="haha")
    r = st.slider("Risk-Free Rate (r)", 0.00, 0.20, 0.06, key="hehe")
    n_steps = st.slider("Time Steps", 50, 365, 252, key="giwhf")
    n_paths = st.slider("Simulations", 1000, 50000, 10000, step=1000,key="hehfe")
    
    # --- Convergence Plot ---
    st.subheader("📈 Convergence Plot")
    def plot_convergence():
        path_range = range(1000, 20001, 2000)
        prices = []
        for n in path_range:
            S_tmp = simulate_paths(S, T, r, sigma, n_steps, n)
            final_tmp = S_tmp[:, -1]
            barrier_tmp = np.any(S_tmp >= barrier, axis=1) if "up" in barrier_type else np.any(S_tmp <= barrier, axis=1)
            valid_tmp = barrier_tmp
            payoff_tmp = np.maximum(final_tmp - K, 0) if option_kind == "call" else np.maximum(K - final_tmp, 0)
            price_tmp = np.exp(-r * T) * np.mean(payoff_tmp * valid_tmp)
            prices.append(price_tmp)

        fig2, ax2 = plt.subplots(figsize=(12,6))
        ax2.plot(path_range, prices, marker='o')
        ax2.set_title("Convergence of Option Price")
        ax2.set_xlabel("Number of Paths")
        ax2.set_ylabel("Estimated Price")
        st.pyplot(fig2)

    plot_convergence()



