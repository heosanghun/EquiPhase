import numpy as np
import torch, math
PI = math.pi
N = 256
c = torch.linspace(-PI, PI, N + 1)[:-1]
P, S = torch.meshgrid(c, c, indexing='ij')

# Let V = sin(P) * cos(S)
V_theta_grid = torch.sin(P) * torch.cos(S)
V_theta_grid = V_theta_grid - V_theta_grid.mean()

S_phi = -torch.cos(P) * torch.cos(S)
S_psi = torch.sin(P) * torch.sin(S)

S_phi = S_phi.numpy()
S_psi = S_psi.numpy()

dx = 2 * PI / N
dy = 2 * PI / N

div_S = (np.roll(S_phi, -1, axis=0) - np.roll(S_phi, 1, axis=0)) / (2 * dx) + \
        (np.roll(S_psi, -1, axis=1) - np.roll(S_psi, 1, axis=1)) / (2 * dy)
b = -div_S.flatten()
b = b - np.mean(b)

N2 = N * N
from scipy.sparse import diags
from scipy.sparse.linalg import cg
main_diag = -4.0 / (dx * dy) * np.ones(N2)
off_diag_x = 1.0 / (dx * dx) * np.ones(N2)
off_diag_y = 1.0 / (dy * dy) * np.ones(N2)
for i in range(1, N): off_diag_x[i * N - 1] = 0.0

L = diags([main_diag, off_diag_x[1:], off_diag_x[1:], off_diag_y[N:], off_diag_y[N:]], [0, -1, 1, -N, N], shape=(N2, N2), format='lil')
for i in range(N):
    L[i, i + N*(N-1)] = 1.0 / (dy * dy)
    L[i + N*(N-1), i] = 1.0 / (dy * dy)
    L[i*N, i*N + N - 1] = 1.0 / (dx * dx)
    L[i*N + N - 1, i*N] = 1.0 / (dx * dx)
L = L.tocsr()

V_P2_flat, info = cg(L, b, rtol=1e-10, maxiter=10000)
V_P2 = V_P2_flat.reshape(N, N)
V_P2 = V_P2 - np.mean(V_P2)

corr_p2 = np.corrcoef(V_theta_grid.numpy().flatten(), V_P2.flatten())[0, 1]
print('Corr:', corr_p2)
