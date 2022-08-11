import numpy as np
import yaml
import statsmodels.api as sm
import pandas as pd
from plot_utilities import *

yaml_file = open("parameters.yml")
parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)["parameters"]
T = parsed_yaml_file['T']
n = parsed_yaml_file['n']
m = parsed_yaml_file['m']
p = parsed_yaml_file['p']
sample_n = parsed_yaml_file['sample_n']

def CtrlTask(s, random_seed=326, x0=1, T=T):
    #This function returns the co-design matrix, Phi.
    #S is not related to it so we assume it is 0 in this function.

    #T: total time steps
    #n: x-dim
    #m: u-dim
    #p: s-dim

    np.random.seed(random_seed)
    # s = np.zeros((p*T,1))

    A = np.random.rand(n,n) + 0.1
    B = np.random.rand(n,m) + 0.1
    C = np.random.rand(n,p) + 0.1

    M_list = []
    N_list = []

    for i in range(T):
        if i == 0:
            M = np.concatenate((B, np.zeros((n, m*(T-1)))), axis=1)
            N = np.concatenate((C, np.zeros((n, p*(T-1)))), axis=1)
            M_list.append(M)
            N_list.append(N)
            continue

        _ = np.dot(A, M_list[-1][:, :m]).reshape((n, m))
        __ = np.dot(A, N_list[-1][:, :p]).reshape((n, p))

        M = np.concatenate((_, M_list[-1][:, :m*(T-1)]), axis=1)
        N = np.concatenate((__, N_list[-1][:, :p*(T-1)]), axis=1)
        M_list.append(M)
        N_list.append(N)

    ### Q R must be PSD
    Q = np.random.rand(n,n)
    Q = Q.T @ Q
    R = np.random.rand(m,m)
    R = R.T @ R

    s = np.array(s)

    # K = np.zeros((m*T, m*T)) + R
    K = np.kron(np.eye(T,dtype=int), R)
    k = np.zeros((m*T,1))
    L = np.zeros((m*T, p*T))

    for t in range(T):
        # K
        _ = np.dot(M_list[t].T, Q)
        _ = np.dot(_, M_list[t])
        K += _

        #k
        __ = np.dot(M_list[t].T, Q)
        Nts = np.dot(N_list[t], s)
        A_pow_t1_x0 = np.dot(np.linalg.matrix_power(A, t+1), x0)
        __ = np.dot(__, (A_pow_t1_x0 + Nts) )
        k += __

        #L
        ___ = np.dot(M_list[t].T, Q)
        ___ = np.dot(___, N_list[t])
        L += ___

    u_opt = -1 * np.dot(np.linalg.inv(K), k)

    Phi = np.dot(L.T, np.linalg.inv(K))
    Phi = np.dot(Phi, L)

    # xt
    x_list = [np.array(x0)]

    for t in range(0, T-1):
        xt = np.dot(np.linalg.matrix_power(A, t+1), x0)
        xt += np.dot(M_list[t], u_opt)
        xt += np.dot(N_list[t], s)
        x_list.append(xt)

    cost = u_opt.T @ K @ u_opt + 2 * k.T @ u_opt
    for t in range(0, T):
        const = np.dot(np.linalg.matrix_power(A, t+1), x0) + N_list[t] @ s
        cost += const.T @ Q @ const

    return Phi, float(cost)


def CtrlTaskIndentity(s, x0=1, T=T):
    #This function returns the co-design matrix, Phi.
    #S is not related to it so we assume it is 0 in this function.

    #T: total time steps
    #n: x-dim
    #m: u-dim
    #p: s-dim

    # s = np.zeros((p*T,1))
    s = s.T

    A = np.eye(n, dtype=np.float64)
    B = - np.eye(m, dtype=np.float64)
    C = np.eye(p, dtype=np.float64)

    M_list = []
    N_list = []

    for i in range(T):
        if i == 0:
            M = np.concatenate((B, np.zeros((n, m*(T-1)))), axis=1)
            N = np.concatenate((C, np.zeros((n, p*(T-1)))), axis=1)
            M_list.append(M)
            N_list.append(N)
            continue

        _ = np.dot(A, M_list[-1][:, :m]).reshape((n, m))
        __ = np.dot(A, N_list[-1][:, :p]).reshape((n, p))

        M = np.concatenate((_, M_list[-1][:, :m*(T-1)]), axis=1)
        N = np.concatenate((__, N_list[-1][:, :p*(T-1)]), axis=1)
        M_list.append(M)
        N_list.append(N)

    # print(M_list[0:2], M_list[0].shape)

    ### Q R must be PSD
    Q = np.eye(m, dtype=np.float64)
    R = np.eye(m, dtype=np.float64)

    # K = np.zeros((m*T, m*T)) + R
    K = np.kron(np.eye(T,dtype=int), R)
    k = np.zeros((m*T, 1))
    L = np.zeros((m*T, p*T))
    # print(K, K.shape, T)

    for t in range(T):
        # K
        _ = np.dot(M_list[t].T, Q)
        _ = np.dot(_, M_list[t])
        K += _

        #k
        __ = np.dot(M_list[t].T, Q)
        # print((N_list[t].shape, s.shape))
        Nts = np.dot(N_list[t], s)
        A_pow_t1_x0 = np.dot(np.linalg.matrix_power(A, t+1), x0)
        __ = np.dot(__, (A_pow_t1_x0 + Nts) )
        k += __

        #L
        ___ = np.dot(M_list[t].T, Q)
        ___ = np.dot(___, N_list[t])
        L += ___

    u_opt = -1 * np.dot(np.linalg.inv(K), k)

    Phi = np.dot(L.T, np.linalg.inv(K))
    Phi = np.dot(Phi, L)
    
    # xt
    x_list = [np.array(x0)]
    # x_list_ = [x0]

    for t in range(0, T):
        xtplus1 = np.dot(np.linalg.matrix_power(A, t+1), x0)
        xtplus1 += np.dot(M_list[t], u_opt)
        xtplus1 += np.dot(N_list[t], s)
        x_list.append(xtplus1)

        # x = np.dot(A, x_list_[-1]) + np.dot(B, u_opt[t]) + np.dot(C, s[t])
        # x_list_.append(x)

    cost = u_opt.T @ K @ u_opt + 2 * k.T @ u_opt
    for t in range(0, T):
        const = np.dot(np.linalg.matrix_power(A, t+1), x0) + N_list[t] @ s
        cost += const.T @ Q @ const

    return Phi, float(cost)


def ConstaintedCtrlTaskParas(random_seed = 326, T=T):
    #This function returns the parameters (A, B, C, Q, R).
    #S is not related to it so we assume it is 0 in this function.

    #T: total time steps
    #n: x-dim
    #m: u-dim
    #p: s-dim

    np.random.seed(random_seed)
    s = np.zeros((p*T,1))

    A = np.random.rand(n,n) + 0.1
    B = np.random.rand(n,m) + 0.1
    C = np.random.rand(n,p) + 0.1

    Q = np.random.rand(n,n) + 0.2
    R = np.random.rand(m,m) + 0.2
    Q = Q.T @ Q
    R = R.T @ R

    return A, B, C, Q, R


### Generate ARIMA time series
def ARIMA_gen(random_seed = 326, sample_n = sample_n):
    # mu, alpha, beta, gamma are parameters for ARIMA.
    # Upper case vars represent vectors.
    # sample_n is the number of samples.
    # Returns S the timeseries data matrix.
    np.random.seed(random_seed)

    MU = np.random.rand(sample_n,1)
    ALPLA =  np.random.rand(sample_n,1)
    BETA = np.random.rand(sample_n,1)
    SIGMA = 0.1 * np.random.rand(sample_n,1)

    S = np.zeros((MU.shape[0], p*T))

    for i in range(MU.shape[0]):
        mu = MU[i]
        alpha = ALPLA[i]
        beta = BETA[i]
        sigma = SIGMA[i]
        w = np.random.randn(T) * sigma
        s = np.zeros(T) + 1 ### S0=1

        for t in range(1, len(w)):
            s[t] = mu + alpha * s[t-1] + w[t] + beta * w[t-1]
        
        S[i, :]=s

    return S, MU, ALPLA, BETA, SIGMA

if __name__ == '__main__':
    # Phi = CtrlTask()
    Phi, cost = CtrlTaskIndentity(s=np.zeros((1,p*T)))
    S = ARIMA_gen()
    # print(Phi.shape, S.shape)

