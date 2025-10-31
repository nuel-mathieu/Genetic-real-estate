import numpy as np
from scipy.interpolate import interp1d, CubicSpline
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from plotting import plot_portfolio
from opti import *

# Situation parameters
age = 26
R = 5000 # Monthly revenue
C = 3000 # Monthly costs
L = 1300 # Monthly rent
Pfi = 40000 # Financial investment initial

# Optimization parameters
max_age = 60

# Economic parameters
notarial_fees = 0.08 # Notarial fees on property price
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
    N = notarial_fees * I
    A = min(A * (pf_financial_assets[t_invest + 1] - N), I) #Down payment
    # Loan not granted if down payment is less than 10% of property price
    if A < 0.1 * I:
        return None
    # Not enough financial assets to pay notarial fees + down payment
    if pf_financial_assets[t_invest + 1] < A + N:
        return None
    # On purchase month
    financial_assets.append(pf_financial_assets[t_invest + 1] - A - N)
    E = I - A
    # # If notarial fees exceed financial assets, adjust down payment
    # if pf_financial_assets[t_invest + 1] < N:
    #     A = 0.0
    #     financial_assets.append(0)
    #     E = I + N - pf_financial_assets[t_invest + 1]
    # else:
    #     A = min(A * (pf_financial_assets[t_invest + 1] - N), I)
    #     financial_assets.append(pf_financial_assets[t_invest + 1] - A - N)
    #     E = I - A
    #print(f"A={A}, N={N}, E={E}, I={I}")
    immo_assets.append(I)
    loan_amounts.append(E)
    M = M * (R - C) # Loan monthly payment
    F = R - C - M # Monthly financial investment after buying property
    legal, rate, duration = find_rate_and_duration(M, E)
    if not legal or t_invest + duration > total_years * 12 or duration < 0:
        #print(f"Can't afford loan : Duration = {duration}, rate = {rate}, M = {M}, E = {E}")
        return None
    #print(f"Loan granted : Duration = {duration}, rate = {rate*12*100:.2f}%, M = {M:.2f}€, E = {E:.2f}€")
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

def net_worth(M, t_invest, A, I, target_age=50):
    """Retourne le patrimoine net final"""
    res = get_portfolio_time_series(M, int(t_invest), A, I)
    if res is None:
        return -1e10  # pénalité forte si prêt impossible
    f, i, l, E, duration, rate = res
    target_index = -1 + int((target_age - max_age) * 12)
    return f[target_index] + i[target_index] - l[target_index]

def main():
    layout = """
    AAA
    AAA
    BCD
    EFG
    """
    fig, axes = plt.subplot_mosaic(layout, figsize=(12, 8))
    plt.subplots_adjust(hspace=0.5)

    # Initial parameters
    init_params = {
        "M": 0.8,
        "t_invest": 12 * 2,  # 2 ans
        "A": 0.8,
        "I": 150000
    }

    bounds = {
        "M": (0.0, 1.0),
        "t_invest": (0, (max_age - age) * 12),
        "A": (0.0, 1.0),
        "I": (100000, 400000)
    }

    history = {
        "value": [],
        "M": [],
        "t_invest": [],
        "E": [],
        "I": [],
        "duration": [],
        "rate": []
    }

    # Compute initial portfolio
    res = get_portfolio_time_series(
        M=init_params["M"],
        t_invest=init_params["t_invest"],
        A=init_params["A"],
        I=init_params["I"]
    )

    if res is not None:
        f, i, l, E, duration, rate = res
        value = f[-1] + i[-1] - l[-1]
        history["value"].append(value)
        history["M"].append(init_params["M"] * (R - C))
        history["t_invest"].append(init_params["t_invest"] / 12)
        history["E"].append(E)
        history["I"].append(init_params["I"])
        history["duration"].append(duration / 12)
        history["rate"].append(rate * 12 * 100)

        plot_portfolio(axes, f, i, l, ages, pf_financial_assets, history)

    # --- Create sliders ---
    axcolor = 'lightgoldenrodyellow'
    ax_M = plt.axes([0.15, 0.20, 0.65, 0.03], facecolor=axcolor)
    ax_t = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_A = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=axcolor)
    ax_I = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)

    sM = Slider(ax_M, 'M (% revenu)', *bounds["M"], valinit=init_params["M"])
    sT = Slider(ax_t, 't_invest (mois)', *bounds["t_invest"], valinit=init_params["t_invest"])
    sA = Slider(ax_A, 'Apport (%)', *bounds["A"], valinit=init_params["A"])
    sI = Slider(ax_I, 'Prix immo (€)', *bounds["I"], valinit=init_params["I"])

    # --- Update function ---
    def update(val):
        M = sM.val
        t_invest = sT.val
        A = sA.val
        I = sI.val

        res = get_portfolio_time_series(M, t_invest, A, I)
        if res is None:
            return
        f, i, l, E, duration, rate = res
        value = f[-1] + i[-1] - l[-1]

        history["value"].append(value)
        history["M"].append(M * (R - C))
        history["t_invest"].append(t_invest / 12)
        history["E"].append(E)
        history["I"].append(I)
        history["duration"].append(duration / 12)
        history["rate"].append(rate * 12 * 100)

        # Clear and replot
        for key in axes:
            axes[key].cla()
        plot_portfolio(axes, f, i, l, ages, pf_financial_assets, history)
        fig.canvas.draw_idle()

    # Attach callbacks
    sM.on_changed(update)
    sT.on_changed(update)
    sA.on_changed(update)
    sI.on_changed(update)

    # Reset button
    reset_ax = plt.axes([0.85, 0.05, 0.1, 0.04])
    button = Button(reset_ax, 'Reset', color=axcolor, hovercolor='0.975')

    def reset(event):
        sM.reset()
        sT.reset()
        sA.reset()
        sI.reset()
    button.on_clicked(reset)

    plt.show()

if __name__ == "__main__":
    main()
