import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq


hbar = 1.0
m = 1.0
a = 1.0
V0 = 50.0

#Wave number in free region
def k_free(E):
    return np.sqrt(2*m*E)/hbar

#Wave number inside the barrier
def k_barrier(E):
    if E < V0:
        #tunneling wave
        return 1j*np.sqrt(2*m*(V0-E))/hbar
    else:
        #propagating wave
        return np.sqrt(2*m*(E-V0))/hbar


# Propagation matrix for a region of width a
def P(q, d):
    return np.array([
        [np.cos(q*d), np.sin(q*d)/q],
        [-q*np.sin(q*d), np.cos(q*d)]], dtype=complex)


#Free region propagation matrix
def P_free(E):
    return P(k_free(E), a)


#Barrier region propagation matrix
def P_barrier(E):
    return P(k_barrier(E), a)


#Total transfer matrix for n barriers
def M_tot(E, n):
    #Start with first free region
    M = P_free(E)
    #Multiply matrixes for each barrier and free region
    for _ in range(n):
        M = M @ P_barrier(E) @ P_free(E)
    return M

#Secular equation defining eigenvalues
def secular(E, n):
    return np.real(M_tot(E, n)[0,1])


#Numerical search for eigenvalues
def find_bound_states(n, Emax=5*V0, ngrid=5000):

    Es = np.linspace(1e-6, Emax, ngrid)
    fvals = [secular(E, n) for E in Es]
    roots = []
    #sign changes to locate roots
    for i in range(len(Es)-1):
        if fvals[i]*fvals[i+1] < 0:
            try:
                root = brentq(lambda E: secular(E,n), Es[i], Es[i+1])
                roots.append(root)
            except:
                pass

    return np.array(roots)


def plot_potential_and_levels(n, roots):
    L = (2*n+1)*a
    x = np.linspace(0, L, 1000)

    V = np.zeros_like(x)

    for j in range(n):
        mask = (x >= (2*j+1)*a) & (x < (2*j+2)*a)
        V[mask] = V0

    plt.figure(figsize=(8,4))

    plt.plot(x/L, V/V0, 'k-', lw=2, label="Potential $V/V_0$")

    for E in roots:
        plt.hlines(E/V0, 0, 1, colors='r', linestyles='--', lw=0.5)

    plt.title(f"{n} potential barriers")
    plt.xlabel("x/L")
    plt.ylabel("Energy / $V_0$")

    plt.ylim(0, max(1.2, (max(roots)/V0)*1.2 if len(roots)>0 else 1.2))

    plt.legend()
    plt.show()

results = {}

for n in [50, 51]:

    roots = find_bound_states(n)

    results[n] = roots

    plot_potential_and_levels(n, roots)

results
