R = 36000 #revenu
C = 600 * 12 # consommation
L = 1200 * 12 # loyer
Rf = 1.08 # rendement financier
I = 300000 # prix immo
Ri = 1.03 # rendement immo
N = 15000 # frais notaire
Te = 1.4 # taux immobilier
De = 20 # duree emprunt
A = None # apport


def update_assets_no_immo(financial_assets, immo_assets, loan_amounts):
    financial_invest = R - C - L
    financial_assets.append(financial_assets[-1] * Rf + financial_invest)
    immo_assets.append(0)
    loan_amounts.append(0)

def update_assets_on_immo(financial_assets, immo_assets, loan_amounts):
    global A
    A = min(financial_assets[-1] * Rf - N, I) # apport, prend en compte la rentabilité de l'annee traitee
    financial_assets.append(financial_assets[-1] - A - N) # tout est investi dans l'immo
    immo_assets.append(I)
    loan_amounts.append(I - A)

def update_assets_with_immo(financial_assets, immo_assets, loan_amounts):
    if loan_amounts[-1] > 0:
        loan_payment = (I - A) * Te / De
        loan_amounts.append(loan_amounts[-1] - loan_payment)
    else:
        loan_payment = 0
        loan_amounts.append(0)
    financial_invest = R - C - loan_payment
    financial_assets.append(financial_assets[-1] * Rf + financial_invest)
    immo_assets.append(immo_assets[-1] * Ri)



# Plot a heatmap of total assets (financial + immo - loan)
# with immo purchase year on x-axis and current year on y-axis

def total_assets(financial_assets, immo_assets, loan_amounts):
    return [f + i - l for f, i, l in zip(financial_assets, immo_assets, loan_amounts)]

def simulate_years(purchase_year, total_years):
    financial_assets = [0]
    immo_assets = [0]
    loan_amounts = [0]

    for year in range(total_years):
        if year < purchase_year:
            update_assets_no_immo(financial_assets, immo_assets, loan_amounts)
        elif year == purchase_year:
            update_assets_on_immo(financial_assets, immo_assets, loan_amounts)
        else:
            update_assets_with_immo(financial_assets, immo_assets, loan_amounts)

    return total_assets(financial_assets, immo_assets, loan_amounts)

def simulate_all_years(total_years):
    all_totals = []
    for purchase_year in range(total_years):
        totals = simulate_years(purchase_year, total_years)
        all_totals.append(totals)
    return all_totals

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    total_years = 40
    all_totals = simulate_all_years(total_years)
    all_totals = np.array(all_totals).T

    plt.imshow(all_totals, origin='lower', aspect='auto', extent=[0, total_years-1, 0, total_years-1])
    plt.colorbar(label='Total Assets')
    plt.xlabel('Year of Property Purchase')
    plt.ylabel('Current Year')
    plt.title('Total Assets Over Time with Property Purchase Timing')
    plt.show()
