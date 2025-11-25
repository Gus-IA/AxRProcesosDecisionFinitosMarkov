probas_transiciones = [
    [
        [0.7, 0.3, 0.0],
        [1.0, 0.0, 0.0],
        [0.8, 0.2, 0.0],
    ],  # p(s0| s0, a0), p(s0| s0, a1), p(s0| s0, a2)
    [
        [0.0, 1.0, 0.0],
        None,
        [0.0, 0.0, 1.0],
    ],  # p(s1| s1, a0), p(s1| s1, a1), p(s1| s1, a2)
    [None, [0.8, 0.1, 0.1], None],  # p(s2| s2, a0), p(s2| s2, a1), p(s2| s2, a2)
]

recompensas = [
    [[+10, 0, 0], [0, 0, 0], [0, 0, 0]],  # r(s0, s0, a0), r(s0, s0, a1), r(s0, s0, a2)
    [[0, 0, 0], [0, 0, 0], [0, 0, -50]],  # r(s1, s1, a0), r(s1, s1, a1), r(s1, s1, a2)
    [[0, 0, 0], [+40, 0, 0], [0, 0, 0]],  # r(s2, s2, a0), r(s2, s2, a1), r(s2, s2, a2)
]

acciones = [[0, 1, 2], [0, 2], [1]]

import numpy as np

# inicializamos q(s, a)

q = np.full((3, 3), -np.inf)  # -np.inf para acciones imposibles
for estado, accion in enumerate(acciones):
    q[estado, accion] = 0.0

print(q)

gamma = 0.90  # factor de descuento

# estimamos q* aplicando la eq. de Bellman de optimalidad

for iteration in range(100):
    q_prev = q.copy()
    for s in range(3):
        for a in acciones[s]:
            q[s, a] = np.sum(
                [
                    probas_transiciones[s][a][sp]
                    * (recompensas[s][a][sp] + gamma * np.max(q_prev[sp]))
                    for sp in range(3)
                ]
            )

print(q)

# política óptima

pi = np.argmax(q, axis=1)

print(pi)

gamma = 0.95  # factor de descuento

for iteration in range(50):
    q_prev = q.copy()
    for s in range(3):
        for a in acciones[s]:
            q[s, a] = np.sum(
                [
                    probas_transiciones[s][a][sp]
                    * (recompensas[s][a][sp] + gamma * np.max(q_prev[sp]))
                    for sp in range(3)
                ]
            )

# política óptima

np.argmax(q, axis=1)
