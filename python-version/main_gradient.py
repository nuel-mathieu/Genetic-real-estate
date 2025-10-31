import numpy as np
from scipy.interpolate import interp1d, CubicSpline
import matplotlib.pyplot as plt
from plotting import plot_portfolio
from opti import gradient

# Situation parameters
age = 26
R = 2500 # Monthly revenue
C = 1000 # Monthly costs
L = 1300 # Monthly rent
Pfi = 15000 # Financial investment initial

# Optimization parameters
max_age = 60

# Economic parameters
notarial_fees = 0.1 # Notarial fees on property price
Rf = 1.06**(1/12) # Financial return
Ri = 1.033**(1/12) # Real estate return
# Loan rates
loan_durations = np.array([7 * 12, 10 * 12, 15 * 12, 20 * 12, 25 * 12])  # in months
loan_rates_raw = np.array([3 / 100 / 12, 3.05 / 100 / 12, 3.18 / 100 / 12, 3.24 / 100 / 12, 3.38 / 100 / 12])  # monthly rates
loan_rates = CubicSpline(loan_durations, loan_rates_raw)

# Pure financial assets time series
total_years = max_age - age
ages = [age + month / 12 for month in range(total_years * 12 + 1)]
pf_financial_assets = [Pfi]
for month in range(total_years * 12):
    financial_invest = R - C - L
    pf_financial_assets.append(pf_financial_assets[-1] * Rf + financial_invest)

def find_rate_and_duration(M, E):
    """
    M : Monthly payment available (%(R - C))
    E : Loan amount
    Returns (rate, duration) tuple for the loan
    """
    if E == 0:
        return True, 0.0, 0
    duration = np.linspace(min(loan_durations), max(loan_durations), 100)
    M_duration = [E * loan_rates(N) / ( 1 - (1 + loan_rates(N))**(-N))
                  for N in duration]
    f = interp1d(M_duration, duration, fill_value="extrapolate")
    duration = f(M)
    duration = np.ceil(duration).astype(int)
    rate = loan_rates(duration)
    return duration <= 25 * 12, rate, duration

def get_portfolio_time_series(M, t_invest, A, I):
    """
    M : Monthly loan payment (%(R - C))
    t_invest : Month to buy property
    A : Down payment (%Pf=Financial assets at t_invest)
    I : Property price
    """
    t_invest = int(t_invest)
    financial_assets = pf_financial_assets[:t_invest]
    immo_assets = [0] * t_invest
    loan_amounts = [0] * t_invest
    # On purchase month
    N = notarial_fees * I
    A = min(A * (pf_financial_assets[t_invest + 1] - N), I)
    financial_assets.append(pf_financial_assets[t_invest + 1] - A - N)
    immo_assets.append(I)
    E = I - A
    loan_amounts.append(E)
    M = M * (R - C) # Loan monthly payment
    F = R - C - M # Monthly financial investment after buying property
    legal, rate, duration = find_rate_and_duration(M, E)
    if not legal:
        print(f"Can't afford loan : Duration = {duration}, rate = {rate}, M = {M}, E = {E}")
        return None
    # After purchase month
    for month in range(t_invest + 1, t_invest + 1 + duration):
        financial_assets.append(financial_assets[-1] * Rf + F)
        immo_assets.append(immo_assets[-1] * Ri)
        loan_amounts.append(loan_amounts[-1] - min(M, loan_amounts[-1]))
    # After loan is paid off
    F = R - C
    for month in range(t_invest + 1 + duration, total_years * 12 + 1):
        financial_assets.append(financial_assets[-1] * Rf + F)
        immo_assets.append(immo_assets[-1] * Ri)
        loan_amounts.append(0)
    return financial_assets, immo_assets, loan_amounts, E, duration, rate

def net_worth(M, t_invest, A, I):
    """Retourne le patrimoine net final"""
    res = get_portfolio_time_series(M, int(t_invest), A, I)
    if res is None:
        return -1e10  # pénalité forte si prêt impossible
    f, i, l, E, duration, rate = res
    return f[-1] + i[-1] - l[-1]

def main():
    # Two axis plot
    #fig, ax = plt.subplots(figsize=(10, 6))
    layout = """
    AAA
    AAA
    BCD
    EFG
    """
    #fig, axes = plt.subplots(7, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [10, 1, 1, 1, 1, 1, 1]})
    fig, axes = plt.subplot_mosaic(layout, figsize=(12, 8))
    print(axes)
    plt.subplots_adjust(hspace=0.35)

    x = np.array([0.8, 5 * 12, 0.8, 200000])   # M, t_invest (months), A, I
    lr = np.array([1e-3, 5e1, 1e-1, 1e7])  # taux d’apprentissage différenciés
    n_iter = 10000

    history = {
        "value": [],
        "M": [],
        "t_invest": [],
        "E": [],
        "I": [],
        "duration": [],
        "rate": []
    }

    for it in range(n_iter):
        value = net_worth(*x)
        g = gradient(net_worth, x)
        x = x + lr * g / (np.linalg.norm(g) + 1e-8)  # mise à jour directionnelle
        #x = update_parameters(x, g, lr)
        # bornes
        x[0] = np.clip(x[0], 0., 1.)
        #x[1] = np.clip(x[1], 1, (40 - age) * 12 - 2)
        x[1] = np.clip(x[1], 4 * 12, 4 * 12)
        x[2] = np.clip(x[2], 0., 1.)
        x[3] = np.clip(x[3], 100000, 300000)
        print(f"Iter {it:02d} | M={x[0]:.3f}, t={x[1]:.2f}, A={x[2]:.2f}, I={x[3]:,.0f} → Patrimoine net={value:,.0f}")

        financial_assets, immo_assets, loan_amounts, E, duration, rate = get_portfolio_time_series(
            M=x[0], # 50% of (R - C)
            t_invest=x[1], # Invest after 1 year
            A=x[2], # 80% down payment
            I=x[3] # Property price
        )
        history["value"].append(value)
        history["M"].append(x[0] * (R - C))
        history["t_invest"].append(x[1] / 12)  # conversion en années
        history["E"].append(E)
        history["I"].append(x[3])
        history["duration"].append(duration / 12)  # conversion en années
        history["rate"].append(rate * 12 * 100)  # conversion en taux annuel en %

        plot_portfolio(axes, financial_assets, immo_assets, loan_amounts, ages, pf_financial_assets, history)


if __name__ == "__main__":
    main()
