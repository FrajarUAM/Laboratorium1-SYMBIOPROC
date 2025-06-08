import numpy as np
import matplotlib.pyplot as plt

# Parametry modelu
p1 = 8.8
p2 = 440
p3 = 100

d1 = 1.375e-14
d2 = 1.375e-4
d3 = 3e-5

k1 = 1.925e-4
k2 = 1e5
k3 = 1.5e5

# Model
def model(t, y, config):
    p53, mdmcyto, mdmn, pten = y
    siRNA = config.get("siRNA", False)
    DNA_damage = config.get("DNA_damage", False)
    PTEN_off = config.get("PTEN_off", False)

    siRNA_factor = 0.02 if siRNA else 1.0
    DNA_factor = 1.0 if DNA_damage else 0.1
    PTEN_factor = 0.0 if PTEN_off else 1.0

    dp53_dt = p1 - d1 * p53 * mdmn ** 2
    dmdmcyto_dt = (p2 * siRNA_factor * (p53 ** 4) / ((p53 ** 4) + k2 ** 4)
                   - k1 * (k3 ** 2) / ((k3 ** 2) + pten ** 2) * mdmcyto
                   - d2 * DNA_factor * mdmcyto)
    dmdmn_dt = k1 * (k3 ** 2) / ((k3 ** 2) + pten ** 2) * mdmcyto - d2 * DNA_factor * mdmn
    dpten_dt = p3 * PTEN_factor * (p53 ** 4) / ((p53 ** 4) + k2 ** 4) - d3 * pten

    return np.array([dp53_dt, dmdmcyto_dt, dmdmn_dt, dpten_dt])

# RK4 z krokiem stałym
def rk4_fixed(f, y0, t0, tf, h, config):
    times = [t0]
    results = [y0]
    y = y0.copy()
    t = t0

    while t < tf:
        k1 = f(t, y, config)
        k2 = f(t + h / 2, y + h / 2 * k1, config)
        k3 = f(t + h / 2, y + h / 2 * k2, config)
        k4 = f(t + h, y + h * k3, config)

        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h

        times.append(t)
        results.append(y)

    return np.array(times), np.array(results)

# RK4 adaptacyjny
def rk4_adaptive(f, y0, t0, tf, h0, config, tol=1e-2, h_min=0.01, h_max=5):
    times = [t0]
    results = [y0.copy()]
    t = t0
    y = y0.copy()
    h = h0

    while t < tf:
        if t + h > tf:
            h = tf - t

        k1 = f(t, y, config)
        k2 = f(t + h / 2, y + h / 2 * k1, config)
        k3 = f(t + h / 2, y + h / 2 * k2, config)
        k4 = f(t + h, y + h * k3, config)
        y_rk4 = y + h / 6 * (k1 + 2*k2 + 2*k3 + k4)

        k1s = f(t, y, config)
        k2s = f(t + h, y + h * k1s, config)
        y_rk2 = y + h / 2 * (k1s + k2s)

        error = np.max(np.abs(y_rk4 - y_rk2))
        if error < tol or h <= h_min:
            t += h
            y = y_rk4
            times.append(t)
            results.append(y.copy())

        s = min(2, max(0.5, 0.9 * (tol / error)**0.5)) if error != 0 else 2
        h = max(h_min, min(h * s, h_max))

    return np.array(times), np.array(results)

# Główna część programu
def main():
    # Stabilizacja
    y0 = np.array([90000, 200000, 250000, 550000])
    t0 = 0
    tf = 300000  # minut
    h = 5
    config_base = {"siRNA": False, "DNA_damage": False, "PTEN_off": False}
    times, results = rk4_fixed(model, y0, t0, tf, h, config_base)

    # Wykres stabilizacji
    labels = ["p53", "MDMcyto", "MDMn", "PTEN"]
    index_map = {label: i for i, label in enumerate(labels)}
    plt.figure(figsize=(12, 8))
    for label in labels:
        plt.plot(times / 60, results[:, index_map[label]], label=label, linewidth=2)
    plt.yscale("log")
    plt.title("Stabilizacja modelu (scenariusz podstawowy)")
    plt.xlabel("Czas [h]")
    plt.ylabel("Liczba cząsteczek (log)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("stabilizacja_podstawowy.png", dpi=300)
    plt.close()

    final_values = results[-1]

    # Symulacje 4 scenariuszy
    initial_conditions = final_values
    t0 = 0
    tf = 2880
    h = 0.5

    scenarios = {
        "A_Podstawowy": {"siRNA": False, "DNA_damage": False, "PTEN_off": False},
        "B_Uszkodzenie_DNA": {"siRNA": False, "DNA_damage": True, "PTEN_off": False},
        "C_Nowotwór": {"siRNA": False, "DNA_damage": True, "PTEN_off": True},
        "D_Terapia": {"siRNA": True, "DNA_damage": True, "PTEN_off": True},
    }

    for name, cfg in scenarios.items():
        # Stały krok
        t_fixed, res_fixed = rk4_fixed(model, initial_conditions, t0, tf, h, cfg)
        plt.figure(figsize=(10, 6))
        for label in labels:
            plt.plot(t_fixed / 60, res_fixed[:, index_map[label]], label=label)
        plt.yscale("log")
        plt.title(f"Scenariusz {name} (RK4 stały krok)")
        plt.xlabel("Czas [h]")
        plt.ylabel("Liczba cząsteczek (log)")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f"scenariusz_{name}_stały_krok.png", dpi=300)
        plt.close()

        # Adaptacyjny krok
        t_adapt, res_adapt = rk4_adaptive(model, initial_conditions, t0, tf, h, cfg)
        plt.figure(figsize=(10, 6))
        for label in labels:
            plt.plot(t_adapt / 60, res_adapt[:, index_map[label]], label=label)
        plt.yscale("log")
        plt.title(f"Scenariusz {name} (RK4 adaptacyjny)")
        plt.xlabel("Czas [h]")
        plt.ylabel("Liczba cząsteczek (log)")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f"scenariusz_{name}_adaptacyjny.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    main()
