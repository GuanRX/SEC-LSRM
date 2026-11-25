import torch




def solve_l1l2(W, lambda_):
    # Each column represents the residual of one superpixel.
    norms = torch.linalg.vector_norm(W, dim=0, keepdim=True)
    return W * (1 - lambda_ / norms.clamp_min(torch.finfo(W.dtype).tiny)).clamp_min(0)


def idr(X, k, maxIter, gamma, lambda_, mu=1e-6):
    tol = 1e-7
    rho = 1.1
    max_mu = 1e30

    # Initializing optimization variables
    d, n = X.shape
    e = torch.ones((1, n), dtype=X.dtype, device=X.device)
    I = torch.eye(n, dtype=X.dtype, device=X.device)
    Z = torch.zeros(n, n, device=X.device)
    S = torch.zeros(n, n, device=X.device)
    E = torch.zeros(d, n, device=X.device)
    C = torch.ones(n, n, dtype=X.dtype, device=X.device)
    D = torch.zeros(n, n, dtype=X.dtype, device=X.device)

    XtX = torch.mm(X.t(), X)
    ete = torch.mm(e.t(), e)
    Y1 = torch.zeros(d, n, dtype=X.dtype, device=X.device)
    Y2 = torch.zeros(n, n, dtype=X.dtype, device=X.device)
    Y3 = torch.zeros(1, n, dtype=X.dtype, device=X.device)
    Y4 = torch.zeros(n, n, dtype=X.dtype, device=X.device)

    iter = 0
    while iter < maxIter:

        iter = iter + 1
        A = 2 * I + mu * XtX
        B = 2 * S + mu * (XtX - torch.mm(X.t(), E)) + torch.mm(X.t(), Y1)
        Z = torch.linalg.solve(A, B)
        del A
        del B
        I_C = I - C

        A = 2 * (1 + mu) * I + 2 * gamma * torch.mm(I_C, I_C.t())
        B = 2 * Z + mu * C - Y2 + mu * D - Y4
        S = torch.linalg.solve(A, B, left=False)
        del A
        del B
        S = 0.5 * (S + S.t())
        S = torch.relu(S) #torch.max(S, torch.zeros_like(S))

        A = 2 * gamma * torch.mm(S.t(), S) + mu * (I + ete)
        B = 2 * gamma * torch.mm(S.t(), S) + mu * S + Y2 + mu * ete - torch.mm(e.t(), Y3)
        #Inv_A = torch.linalg.pinv(A)
        #C = torch.mm(Inv_A, B)
        C = torch.linalg.solve(A, B)
        del A
        del B

        A = S + Y4 / mu
        t = torch.diag(A)
        D = A - torch.diag(t)
        tau = 1
        eta = 2 * (k * tau - t.sum()) / n
        d = t + eta / 2
        D = D + torch.diag(d)

        del A
        A = mu * (X - torch.mm(X, Z)) + Y1
        E = solve_l1l2(A / mu, lambda_ / mu)
        leq1 = X - torch.mm(X, Z) - E
        leq2 = S - C
        leq3 = torch.mm(e, C) - e
        leq4 = S - D

        stopC = max([leq1.abs().max(), leq2.abs().max(), leq3.abs().max(), leq4.abs().max()])

        # if iter == 1 or iter % 50 == 0 or stopC < tol:
        #     print(f'iter {iter}, mu={mu:e}, stopALM={stopC:.3e}')

        if stopC < tol:
            break
        else:
            Y1 = Y1 + mu * leq1
            Y2 = Y2 + mu * leq2
            Y3 = Y3 + mu * leq3
            Y4 = Y4 + mu * leq4
            mu = min(max_mu, mu * rho)

    return Z.t(), S.t()


