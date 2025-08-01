PROJECT TITLE: Monte Carlo Based Option Pricing Model

OVERVIEW A Streamlit-based web app to simulate European option pricing using Monte Carlo simulations, with comparisons to Black-Scholes pricing. Useful for finance students, quant developers, and researchers.

FEATURES Real-time stock price data via Yahoo Finance (yfinance) European Call and Put Option pricing using: Monte Carlo simulation Black-Scholes formula American Call and Put Option pricing using: LSMC Method Binomial Tree Method Exotic Call and Put Option pricing using: Asian Option Barrier Option(Up-in and Down-in) Visualization of simulated GBM price paths Intuitive UI with interactive sliders (Streamlit) Volatility estimation from historical data Configurable simulation parameters (strike %, T, N, M, r)

TECH STACK Python 3.9+ Streamlit (UI) NumPy, Pandas, Matplotlib yfinance (stock data) SciPy (norm stats)

INSTALLATION git clone https://github.com/Nitya-sigadapu/monte_carlo-option.git cd monte_carlo-option pip install -r requirements.txt streamlit run app.py

HOW IT WORKS

Pulls historical stock data (e.g., INFOSYS.NS)
Estimates volatility from log returns
Simulates Geometric Brownian Motion paths
Computes discounted expected payoff for options
Compares to analytical pricing
Plots Greeks and price paths
Plot sensitivity plots and add unit tests
SAMPLE RESULTS

EUROPEAN

MONTE CARLO METHOD Put option price : ₹195.88 Call option price : ₹82.62 Greeks (for call) Delta : ₹0.41 Gamma : ₹-.00 Vega : ₹452.87 Theta : ₹-158.50 Rho : ₹288.53

BLACK SCHOLES MODEL Put option price : ₹82.73 Call option price : ₹199.86 Greeks Delta : ₹0.40 Gamma : ₹0.00 Vega : ₹439.66 Theta : ₹-152.94 Rho : ₹281.31

REFERENCES 1.Quant Guild https://youtu.be/-1RYvajksjQ?si=ebRkVJXbGWjZGRPG 2.Very Normal s://youtu.be/Cb-GwN6jhNE?si=UhmXr5HEcZEUy5N3 3.httphttps://github.com/just-krivi/option-pricing-models 4.https://youtu.be/OdWLP8umw3A?si=_q1-a6SdZQjyJ0J4 5.https://youtu.be/r7cn3WS5x9c?si=lceJjzmHTvvGq3IU

AUTHOR Github:https://github.com/Nitya-sigadapu Github:https://github.com/thchrn Github:https://github.com/Prayuktha-Lucky-Reddy

LICENSE This project is licensed under the MIT License

CONTRIBUTORS Feel free to fork, improve, and submit pull requests. All contributions are welcome!
