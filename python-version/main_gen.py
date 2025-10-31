import numpy as np
from scipy.interpolate import interp1d, CubicSpline
import matplotlib.pyplot as plt
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

    # --- paramètres GA ---
    POP_SIZE = 200         # taille de la population
    N_GENERATIONS = 1000    # nombre d’itérations
    MUT_RATE = 0.2         # taux de mutation
    ELITE_FRAC = 0.2       # fraction d’élite conservée

    # --- bornes pour les variables ---
    bounds = {
        "M": (0., 1.),                   # % du (R - C)
        "t_invest": (0, (max_age - age) * 12), # mois avant achat
        #"t_invest": (3 * 12, 3 * 12), # mois avant achat
        "A": (0., 1.),                   # % d’apport
        "I": (250000, 400000),          # prix immobilier
    }

    pop = [random_individual(bounds) for _ in range(POP_SIZE)]

    history = {
        "value": [],
        "M": [],
        "t_invest": [],
        "E": [],
        "I": [],
        "duration": [],
        "rate": []
    }

    for GEN in range(N_GENERATIONS):

        scores = np.array([fitness(ind, net_worth) for ind in pop])
        best_idx = np.argmax(scores)
        best = pop[best_idx]
        best_score = scores[best_idx]

        print(f"Génération {GEN:03d} | Score={best_score:,.0f} | "
              f"M={best[0]:.2f}, t={best[1]/12:.1f} ans, A={best[2]:.2f}, I={best[3]:,.0f}")

        # sélection (élitisme + roulette)
        elite_size = int(ELITE_FRAC * POP_SIZE)
        elite_idx = scores.argsort()[-elite_size:]
        elite = [pop[i] for i in elite_idx]
        probs = (scores - scores.min()) + 1e-6
        probs /= probs.sum()

        new_pop = elite.copy()
        while len(new_pop) < POP_SIZE:
            idx = np.random.choice(len(pop), size=2, p=probs)
            parents = [pop[i] for i in idx]
            child = crossover(*parents)
            child = mutate(child, bounds, MUT_RATE)
            new_pop.append(child)

        pop = new_pop

        #Plotting
        financial_assets, immo_assets, loan_amounts, E, duration, rate = get_portfolio_time_series(
            M=best[0], # 50% of (R - C)
            t_invest=best[1], # Invest after 1 year
            A=best[2], # 80% down payment
            I=best[3] # Property price
        )
        history["value"].append(best_score)
        history["M"].append(best[0] * (R - C))
        history["t_invest"].append(best[1] / 12)  # conversion en années
        history["E"].append(E)
        history["I"].append(best[3])
        history["duration"].append(duration / 12)  # conversion en années
        history["rate"].append(rate * 12 * 100)  # conversion en taux annuel en %

        plot_portfolio(axes, financial_assets, immo_assets, loan_amounts, ages, pf_financial_assets, history)
    plt.show()

if __name__ == "__main__":
    main()
