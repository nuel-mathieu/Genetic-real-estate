import numpy as np
import matplotlib.pyplot as plt


def plot_portfolio(axes, financial_assets, immo_assets, loan_amounts, ages, pf_financial_assets, history):
    for name, ax in axes.items():
            ax.clear()

    ax = axes['A']
    neg_loan = -np.array(loan_amounts)
    ax.stackplot(
        ages,
        financial_assets,
        immo_assets,
        neg_loan,
        labels=["Actifs financiers", "Actifs immobiliers", "Montants des prêts"],
        colors=["#66b3ff", "#99e699", "#ff9999"],
        alpha=0.8
    )
    # --- Ligne noire : actifs financiers purs
    ax.plot(
        ages,
        pf_financial_assets,
        color='black',
        linestyle='--',
        linewidth=2,
        label="Actifs financiers purs"
    )
    # --- Mise en forme
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_title("Évolution du patrimoine avec investissement immobilier")
    ax.set_xlabel("Âge (années)")
    ax.set_ylabel("Valeur (€)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.show(block=False)
    # Parameters plots
    ax_value, ax_t, ax_I, ax_M, ax_duration, ax_rate = axes['B'], axes['C'], axes['D'], axes['E'], axes['F'], axes['G']
    ax_value.plot(history["value"], label="Patrimoine net (€)", color="black")
    ax_t.plot(history["t_invest"], label="t_invest (années)", color="green", alpha=0.5)
    ax_I.plot(history["I"], label="I (prix)", color="red", alpha=0.5)
    ax_I.plot(history["E"], label="E (emprunt)", color="purple", alpha=0.5)
    ax_M.plot(history["M"], label="M (mensualités en €)", color="brown", alpha=0.5)
    ax_duration.plot(history["duration"], label="duration (années)", color="orange", alpha=0.5)
    ax_rate.plot(history["rate"], label="rate (%)", color="blue", alpha=0.5)
    for ax in [ax_value, ax_t, ax_I, ax_M, ax_duration, ax_rate]:
        ax.legend(loc="upper right")
        ax.set_xlabel("Itération")
        ax.grid(True, linestyle=':')
    plt.pause(0.01)