import numpy as np
from scipy.interpolate import interp1d, CubicSpline


# Situation parameters
age = 26
R = 2500 # Monthly revenue
C = 1000 # Monthly costs
L = 1000 # Monthly rent
Pfi = 15000 # Financial investment initial

# Optimization parameters
max_age = 60

# Economic parameters
Rf = 1.06**(1/12) # Financial return
Ri = 1.033**(1/12) # Real estate return
# Loan rates
loan_durations = np.array([7 * 12, 10 * 12, 15 * 12, 20 * 12, 25 * 12])  # in months
loan_rates_raw = np.array([3 / 100 / 12, 3.05 / 100 / 12, 3.18 / 100 / 12, 3.24 / 100 / 12, 3.38 / 100 / 12])  # monthly rates
loan_rates = CubicSpline(loan_durations, loan_rates_raw)

# Pure financial assets time series
total_years = max_age - age
pf_financial_assets = [Pfi]
for month in range(total_years * 12):
    financial_invest = R - C
    pf_financial_assets.append(pf_financial_assets[-1] * Rf + financial_invest)

def find_rate_and_duration(M, E):
    """
    M : Monthly payment available (%(R - C))
    E : Loan amount
    Returns (rate, duration) tuple for the loan
    """
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
    ages = [age + month / 12 for month in range(total_years * 12)]
    financial_assets = pf_financial_assets[:t_invest]
    immo_assets = [0] * t_invest
    loan_amounts = [0] * t_invest
    # On purchase month
    A = min(A * pf_financial_assets[t_invest + 1], I)
    financial_assets.append(pf_financial_assets[t_invest + 1] - A)
    immo_assets.append(I)
    E = I - A
    loan_amounts.append(E)
    M = M * (R - C) # Loan monthly payment
    F = R - C - M # Monthly financial investment after buying property
    legal, rate, duration = find_rate_and_duration(M, E)
    if not legal:
        print("Can't afford loan")
        return None
    # After purchase month
    for month in range(t_invest + 2, t_invest + 2 + duration):
        financial_assets.append(financial_assets[-1] * Rf + F)
        immo_assets.append(immo_assets[-1] * Ri)
        loan_amounts.append(loan_amounts[-1] - min(M, loan_amounts[-1]))
    # After loan is paid off
    F = R - C
    for month in range(t_invest + 3 + duration, total_years * 12):
        financial_assets.append(financial_assets[-1] * Rf + F)
        immo_assets.append(immo_assets[-1] * Ri)
        loan_amounts.append(0)
    return ages, financial_assets, immo_assets, loan_amounts


def main():
    import plotly.graph_objects as go

    print("Hello from immo!")
    ages, financial_assets, immo_assets, loan_amounts = get_portfolio_time_series(
        M=0.8, # 50% of (R - C)
        t_invest=12 * 2, # Invest after 2 years
        A=0.8, # 80% down payment
        I=200000 # Property price
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ages,
        y=financial_assets,
        stackgroup='one',
        name='Actifs financiers'
    ))
    fig.add_trace(go.Scatter(
        x=ages,
        y=immo_assets,
        stackgroup='one',
        name='Actifs immobiliers'
    ))
    fig.add_trace(go.Scatter(
        x=ages,
        y=[-loan_amount for loan_amount in loan_amounts],
        stackgroup='one',
        name='Montants des prêts'
    ))
    # Add pure financial assets as black dashed line
    fig.add_trace(go.Scatter(
        x=ages,
        y=pf_financial_assets[:len(ages)],
        mode='lines',
        line=dict(color='black', dash='dash'),
        name='Actifs financiers purs'
    ))
    fig.update_layout(title="Évolution du patrimoine avec investissement immobilier",
                      yaxis_title="Valeur (€)",
                      xaxis_title="Âge (années)",
                      hovermode='x unified')

    fig.show()

if __name__ == "__main__":
    main()
