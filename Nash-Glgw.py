import numpy as np

# ORIGINAL PREISRASTER AUS DEINER ARBEIT
prices = np.array([
    1.427725, 1.466472, 1.505219, 1.543966, 1.582713,
    1.621460, 1.660207, 1.698954, 1.737701, 1.776448,
    1.815195, 1.853942, 1.892689, 1.931436, 1.970185
])
n_prices = len(prices)


def get_profits(p1_idx, p2_idx):
    p1, p2 = prices[p1_idx], prices[p2_idx]
    mu = 0.25
    a = 2.0
    a0 = 0.0
    c = 1.0

    v1 = (a - p1) / mu
    v2 = (a - p2) / mu
    v0 = a0 / mu

    exp_v = np.exp([v1, v2, v0])
    shares = exp_v / np.sum(exp_v)

    profit1 = shares[0] * (p1 - c)
    profit2 = shares[1] * (p2 - c)
    return profit1, profit2


print("==========================================================================")
# Suchschleife nach ALLEN Nash-Gleichgewichten
found_equilibria = []

for idx1 in range(n_prices):
    for idx2 in range(n_prices):
        prof1, prof2 = get_profits(idx1, idx2)

        # Beste Reaktionen ermitteln
        h1_best_deviation = max([get_profits(i, idx2)[0] for i in range(n_prices)])
        h2_best_deviation = max([get_profits(idx1, j)[1] for j in range(n_prices)])

        # Nash-Bedingung prüfen
        if prof1 >= h1_best_deviation - 1e-5 and prof2 >= h2_best_deviation - 1e-5:
            found_equilibria.append({
                "p1_idx": idx1, "p2_idx": idx2,
                "p1": prices[idx1], "p2": prices[idx2],
                "prof1": prof1, "prof2": prof2
            })

print(f"Gefundene Nash-Gleichgewichte im diskreten Raster: {len(found_equilibria)}\n")
print(f"{'Index (H1, H2)':<15} | {'Preis H1':<10} | {'Preis H2':<10} | {'Profit H1':<12} | {'Profit H2'}")
print("-" * 75)
for eq in found_equilibria:
    print(
        f"({eq['p1_idx']:>2}, {eq['p2_idx']:>2})       | {eq['p1']:<10.4f} | {eq['p2']:<10.4f} | {eq['prof1']:<12.6f} | {eq['prof2']:.6f}")
print("==========================================================================")