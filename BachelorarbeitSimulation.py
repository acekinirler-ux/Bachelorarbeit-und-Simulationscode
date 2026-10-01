import numpy as np
import matplotlib.pyplot as plt
import random


# --- 1. PARAMETER, SEEDS & SYSTEM-INITIALISIERUNG ---

delta = 0.98
max_episodes = 2000000
CONVERGENCE_THRESHOLD = 100000  # Runden exakt gleicher Preise für stabile Konvergenz

# ORIGINAL CALVANO ET AL. (2020) PREISRASTER
prices = np.array([
    1.427725, 1.466472, 1.505219, 1.543966, 1.582713,
    1.621460, 1.660207, 1.698954, 1.737701, 1.776448,
    1.815195, 1.853942, 1.892689, 1.931436, 1.970185
])
n_prices = len(prices)

# Dynamische Generierung von 100 Seeds (1 bis 100)
seeds = list(range(1, 101))
n_runs = len(seeds)

# Datenstrukturen für die finalen statistischen Auswertungen (aus der Evaluationsphase)
all_final_prices_p1 = []
all_final_prices_p2 = []
all_final_profits_p1 = []
all_final_profits_p2 = []
history_convergence_episodes = []

# Listen für das globale zeitliche Tracking über das gesamte Training (für Plots)
global_epsilon_history = []
global_alpha_history = []

# Matrizen zum Speichern der exakten Verläufe in den 100 Testrunden über alle 100 Seeds
eval_prices_p1_matrix = np.zeros((n_runs, 100))
eval_prices_p2_matrix = np.zeros((n_runs, 100))
eval_profits_p1_matrix = np.zeros((n_runs, 100))
eval_profits_p2_matrix = np.zeros((n_runs, 100))



# --- 2. LOGIT-PROFIT-FUNKTION (MARKTMODELL NACH CALVANO) ---

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



# --- 3. BERECHNUNG DER ÖKONOMISCHEN BENCHMARKS (SPIELTHEORIE) ---

nash_price, nash_profit = None, -1
collusive_price, max_joint_profit = None, -1

for idx1 in range(n_prices):
    for idx2 in range(n_prices):
        prof1, prof2 = get_profits(idx1, idx2)
        joint_prof = prof1 + prof2
        if joint_prof > max_joint_profit:
            max_joint_profit = joint_prof
            collusive_price = prices[idx1]

for idx1 in range(n_prices):
    for idx2 in range(n_prices):
        prof1, prof2 = get_profits(idx1, idx2)
        h1_best_deviation = max([get_profits(i, idx2)[0] for i in range(n_prices)])
        h2_best_deviation = max([get_profits(idx1, j)[1] for j in range(n_prices)])

        if prof1 >= h1_best_deviation - 1e-5 and prof2 >= h2_best_deviation - 1e-5:
            nash_price = prices[idx1]
            nash_profit = prof1

# Monopol-Profit pro Spieler (für den Index benötigt)
monopoly_profit_per_player = max_joint_profit / 2


# --- 4. MULTI-RUN SIMULATION (TRAINING & EVALUATION GETRENNT) ---

print(f"Starte {n_runs} Simulationsdurchläufe mit globaler Lernrate & Konvergenzprüfung...\n")

for run_idx, seed in enumerate(seeds):
    random.seed(seed)
    np.random.seed(seed)

    # Matrix-Initialisierung für das Training
    q_table1 = np.zeros((n_prices, n_prices, n_prices))
    q_table2 = np.zeros((n_prices, n_prices, n_prices))

    visit_counts1 = np.zeros((n_prices, n_prices, n_prices))
    visit_counts2 = np.zeros((n_prices, n_prices, n_prices))

    s1, s2 = n_prices // 2, n_prices // 2

    consecutive_stable_rounds = 0
    last_p1_idx, last_p2_idx = -1, -1
    converged_at = max_episodes

    epsilon_run_history = []
    alpha_run_history = []


    # PHASE A: TRAINING (Mit Exploration & Q-Updates)

    for t_idx in range(max_episodes):
        t = t_idx + 1
        current_epsilon = 50000 / (50000 + t)

        if random.uniform(0, 1) < current_epsilon:
            a1, a2 = random.randint(0, n_prices - 1), random.randint(0, n_prices - 1)
        else:
            a1, a2 = np.argmax(q_table1[s1, s2]), np.argmax(q_table2[s1, s2])

        revenue_1, revenue_2 = get_profits(a1, a2)
        new_s1, new_s2 = a1, a2

        # Bellman-Updates
        visit_counts1[s1, s2, a1] += 1
        visit_counts2[s1, s2, a2] += 1
        alpha1 = 1 / (visit_counts1[s1, s2, a1] ** 0.6)
        alpha2 = 1 / (visit_counts2[s1, s2, a2] ** 0.6)

        target1 = revenue_1 + delta * np.max(q_table1[new_s1, new_s2])
        q_table1[s1, s2, a1] += alpha1 * (target1 - q_table1[s1, s2, a1])

        target2 = revenue_2 + delta * np.max(q_table2[new_s1, new_s2])
        q_table2[s1, s2, a2] += alpha2 * (target2 - q_table2[s1, s2, a2])

        s1, s2 = new_s1, new_s2

        # Daten-Tracking fürs Plotten (jede 2000. Runde)
        if t_idx % 2000 == 0:
            epsilon_run_history.append(current_epsilon)
            current_alphas = np.where(visit_counts1 > 0, 1 / (visit_counts1 ** 0.6), 1.0)
            alpha_run_history.append(np.mean(current_alphas))

        # Konvergenzprüfung der gelernten Politik
        if current_epsilon < 0.01:
            pure_exploitation_a1 = np.argmax(q_table1[s1, s2])
            pure_exploitation_a2 = np.argmax(q_table2[s1, s2])

            if pure_exploitation_a1 == last_p1_idx and pure_exploitation_a2 == last_p2_idx:
                consecutive_stable_rounds += 1
            else:
                consecutive_stable_rounds = 0

            last_p1_idx, last_p2_idx = pure_exploitation_a1, pure_exploitation_a2

            if consecutive_stable_rounds >= CONVERGENCE_THRESHOLD:
                converged_at = t
                break

    history_convergence_episodes.append(converged_at)
    global_epsilon_history.append(epsilon_run_history)
    global_alpha_history.append(alpha_run_history)


    # PHASE B: EVALUATION (Einfrieren der Q-Table, Epsilon = 0, Reine Greedy-Policy)

    eval_p1, eval_p2 = [], []
    eval_r1, eval_r2 = [], []
    eval_s1, eval_s2 = s1, s2

    for step in range(100):
        a1_eval = np.argmax(q_table1[eval_s1, eval_s2])
        a2_eval = np.argmax(q_table2[eval_s1, eval_s2])

        rev_1_eval, rev_2_eval = get_profits(a1_eval, a2_eval)

        eval_prices_p1_matrix[run_idx, step] = prices[a1_eval]
        eval_prices_p2_matrix[run_idx, step] = prices[a2_eval]
        eval_profits_p1_matrix[run_idx, step] = rev_1_eval
        eval_profits_p2_matrix[run_idx, step] = rev_2_eval

        eval_p1.append(prices[a1_eval])
        eval_p2.append(prices[a2_eval])
        eval_r1.append(rev_1_eval)
        eval_r2.append(rev_2_eval)

        eval_s1, eval_s2 = a1_eval, a2_eval

    # Durchschnittswerte dieses isolierten Testlaufs berechnen
    run_avg_p = np.mean(eval_p1 + eval_p2)
    run_avg_r = np.mean(eval_r1 + eval_r2)

    all_final_prices_p1.append(np.mean(eval_p1))
    all_final_prices_p2.append(np.mean(eval_p2))
    all_final_profits_p1.append(np.mean(eval_r1))
    all_final_profits_p2.append(np.mean(eval_r2))

    # Lokalen Kollusionsindex berechnen
    run_coll_index = (run_avg_r - nash_profit) / (monopoly_profit_per_player - nash_profit)

    if converged_at < max_episodes:
        print(f" -> Run {run_idx + 1}/{n_runs} (Seed {seed}): Konvergenz bei Runde {converged_at:,}! "
              f"Ø-Preis: {run_avg_p:.4f} €, Kollusionsindex: {run_coll_index:.2%}")
    else:
        print(f" -> Run {run_idx + 1}/{n_runs} (Seed {seed}): Maximum erreicht. "
              f"Ø-Preis: {run_avg_p:.4f} €, Kollusionsindex: {run_coll_index:.2%}")


# --- 5. ERWEITERTE STATISTISCHE KENNZAHLEN-AUSGABE ---

pooled_prices = all_final_prices_p1 + all_final_prices_p2
pooled_profits = all_final_profits_p1 + all_final_profits_p2

mean_price = np.mean(pooled_prices)
sd_price = np.std(pooled_prices)

mean_profit = np.mean(pooled_profits)
sd_profit = np.std(pooled_profits)

global_collusion_index = (mean_profit - nash_profit) / (monopoly_profit_per_player - nash_profit)

print("\n" + "=" * 85)
print(
    f"{'Metrik (Aggregiert über alle 100 Seeds)':<40} | {'Nash (Wettbewerb)':<17} | {'KI-Simulation':<17} | {'Monopol (Kollusion)'}")
print("-" * 85)
print(f"{'Durchschnitts-Preis (Mean)':<40} | {nash_price:<17.4f} | {mean_price:<17.4f} | {collusive_price:.4f}")
print(f"{'Preis-Standardabweichung (SD)':<40} | {'0.0000':<17} | {sd_price:<17.4f} | {'0.0000'}")
print("-" * 85)
print(
    f"{'Durchschnitts-Profit pro Spieler (Mean)':<40} | {nash_profit:<17.4f} | {mean_profit:<17.4f} | {monopoly_profit_per_player:.4f}")
print(f"{'Profit-Standardabweichung (SD)':<40} | {'0.0000':<17} | {sd_profit:<17.4f} | {'0.0000'}")
print("-" * 85)
print(f"{'Ø Konvergenz-Runde (Episoden)':<40} | {'-':<17} | {int(np.mean(history_convergence_episodes)):<17,} | {'-'}")
print(f"{'Kollusionsindex (Profit Gain Delta)':<40} | {'0.00%':<17} | {global_collusion_index:<17.2%} | {'100.00%'}")
print("=" * 85)


# --- 6. VISUALISIERUNG (4 SEPARATE GRAPHIKEN) ---

x_eval_steps = np.arange(1, 101)

# --- GRAFIK 1: DURCHSCHNITTSPREISE (Aggregiert über alle Seeds) ---
plt.figure(figsize=(8, 5))
mean_eval_p1 = np.mean(eval_prices_p1_matrix, axis=0)
mean_eval_p2 = np.mean(eval_prices_p2_matrix, axis=0)

plt.plot(x_eval_steps, mean_eval_p1, label='Algorithmus 1', color='blue', linewidth=2)
plt.plot(x_eval_steps, mean_eval_p2, label='Algorithmus 2', color='orange', linewidth=2,
         linestyle='--')
plt.axhline(y=collusive_price, color='red', linestyle='--', linewidth=1.5, label='Monopolpreis')
plt.axhline(y=nash_price, color='black', linestyle=':', linewidth=1.5, label='Kompetitiver Benchmark')

plt.title("Durchschnittspreise in der Evaluationsphase (ε = 0)", fontsize=12, fontweight='bold')
plt.xlabel("Evaluationsrunden (Testphase)", fontsize=10)
plt.ylabel("Preise", fontsize=10)
plt.ylim(1.35, 2.05)
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# --- GRAFIK 2: DURCHSCHNITTSPROFIT (Aggregiert über alle Seeds) ---
plt.figure(figsize=(8, 5))
mean_eval_r1 = np.mean(eval_profits_p1_matrix, axis=0)
mean_eval_r2 = np.mean(eval_profits_p2_matrix, axis=0)
mean_joint_eval_profit = (mean_eval_r1 + mean_eval_r2) / 2

plt.plot(x_eval_steps, mean_joint_eval_profit, label='Durchschnittsprofite in der Evaluationsphase (ε = 0)', color='green', linewidth=2)
plt.axhline(y=monopoly_profit_per_player, color='red', linestyle='--', linewidth=1.5, label='Monopol-Profit')
plt.axhline(y=nash_profit, color='black', linestyle=':', linewidth=1.5, label='Benchmark-Profit')

plt.title("Durchschnittlicher Profit in der Evaluationsphase (ε = 0)", fontsize=12, fontweight='bold')
plt.xlabel("Evaluationsrunden (Testphase)", fontsize=10)
plt.ylabel("Profite", fontsize=10)
plt.ylim(nash_profit - 0.02, monopoly_profit_per_player + 0.02)
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# --- Berechnen der X-Achsen-Skalierung für die Verläufe ---
longest_run_idx = np.argmax(history_convergence_episodes)
# Sicherer Zuschnitt auf die minimale Länge aller aufgezeichneten Runs, um Inhomogenitäten abzufangen
min_tracked_points = min(len(run) for run in global_alpha_history)
x_ticks_eps = np.linspace(0, min_tracked_points - 1, 5, dtype=int)
x_labels_eps = [f"{int((pos * 2000) / 1000)}k" for pos in x_ticks_eps]

# --- GRAFIK 3: EPSILON-VERLAUF (Globale Exploration) ---
plt.figure(figsize=(8, 5))
# Kürzen des Epsilon-Verlaufs für eine homogene Darstellung im Plot
truncated_epsilon_history = global_epsilon_history[longest_run_idx][:min_tracked_points]
plt.plot(truncated_epsilon_history, color='purple', linewidth=2.5, label='Reduktion der Explorationsrate (ε)')
plt.title("Epsilon-Verlauf über die Trainingszeit", fontsize=12, fontweight='bold')
plt.xlabel("Episoden (Trainingsphase)", fontsize=10)
plt.ylabel("Explorationsrate (ε)", fontsize=10)
plt.xticks(ticks=x_ticks_eps, labels=x_labels_eps)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')
plt.tight_layout()

# --- GRAFIK 4: ALPHA-VERLAUF (Gesamte Matrix) ---
plt.figure(figsize=(8, 5))
# Jede Historie wird auf die minimale gemeinsame Länge gekürzt, damit np.mean() fehlerfrei funktioniert
homogeneous_alpha_history = [run[:min_tracked_points] for run in global_alpha_history]
mean_alpha_trajectory = np.mean(homogeneous_alpha_history, axis=0)

plt.plot(mean_alpha_trajectory, color='darkred', linewidth=2.5, label='Ø Lernrate (α)')
plt.title("Durchschnittlicher Alpha-Verlauf über die Gesamtmatrix", fontsize=12, fontweight='bold')
plt.xlabel("Episoden (Trainingsphase)", fontsize=10)
plt.ylabel("Lernrate (α)", fontsize=10)
plt.xticks(ticks=x_ticks_eps, labels=x_labels_eps)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')
plt.tight_layout()

# Alle Fenster separat anzeigen
plt.show()