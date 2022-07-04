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

def CtrlTask(random_seed=326, x0=1, T=T):
    #This function returns the co-design matrix, Phi.
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

    K = np.zeros((m*T, m*T)) + R
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

    # print("Q", np.allclose(np.linalg.cholesky(Q) @ np.linalg.cholesky(Q).T, Q))
    # print("R", np.allclose(np.linalg.cholesky(R) @ np.linalg.cholesky(R).T, R))
    # print("K", np.allclose(np.linalg.cholesky(K) @ np.linalg.cholesky(K).T, K))
    # print("K sym", np.allclose(K.T, K))
    # print("K-1", np.allclose(np.linalg.cholesky(np.linalg.inv(K)) @ np.linalg.cholesky(np.linalg.inv(K)).T, np.linalg.inv(K)))
    # val, vec = np.linalg.eigh(np.linalg.inv(K))
    # print("K-1 val:", val[0:10], min(val), max(val))
    # print(np.allclose(np.linalg.inv(K)@K, np.eye(K.shape[0], K.shape[1])))
    # print(np.linalg.inv(K)@K)
    # print("Phi", np.allclose(np.linalg.cholesky(Phi) @ np.linalg.cholesky(Phi).T, Phi))

    # xt
    x_list = [np.array(x0)]
    x_list_ = [x0]

    for t in range(0, T-1):
        xt = np.dot(np.linalg.matrix_power(A, t+1), x0)
        xt += np.dot(M_list[t], u_opt)
        xt += np.dot(N_list[t], s)
        x_list.append(xt)

        x = np.dot(A, x_list_[-1]) + np.dot(B, u_opt[t]) + np.dot(C, s[t])
        x_list_.append(x)

    cost = u_opt.T @ K @ u_opt + 2 * k.T @ u_opt
    for t in range(0, T-1):
        const = np.dot(np.linalg.matrix_power(A, t+1), x0) + N_list[t] @ s
        cost += const.T @ Q @ const

    return Phi, float(cost)

def ConstaintedCtrlTaskParas(random_seed = 326, T=T):
    #This function returns the co-design matrix, Phi.
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

def SARIMA_gen(sample_n = sample_n, T = T, random_seed = 326, save_path="./"):
    np.random.seed(random_seed)

    date = pd.date_range('2010-01-01', freq='H', periods=T)
    coeff_batch = []
    time_series_batch = []

    for ii in range(sample_n):
        x = np.random.randint(20, size=(T, 1))
        df = pd.DataFrame(x, columns=['data'], index=date)
        mod = sm.tsa.arima.ARIMA(df, order=(1, 0, 1), seasonal_order=(2, 1, 0, 24))
        model_fit = mod.fit()
        # yhat = model_fit.predict()
        yhat = model_fit.forecast(T)
        # yhat.plot()
        # plt.show()
        ### parameters
        # ar.L1   
        # ma.L1   
        # ar.S.L24
        # ar.S.L48
        # sigma2  
        coeff = model_fit.params.to_numpy(dtype=np.float64)
        coeff_batch.append(coeff)
        time_series_batch.append(yhat)
        print(coeff)
        # print(model_fit.summary())
        # print(model_fit.params)
        print(yhat.head())

    coeff_batch = np.array(coeff_batch)
    time_series_batch = np.array(time_series_batch)
    print(coeff_batch.shape)
    print(time_series_batch.shape)
    np.save(save_path + "coeff_batch", coeff_batch)
    np.save(save_path + "time_series_batch", time_series_batch)


def ARIMA_gen(random_seed = 326,
    sample_n = sample_n):
    # mu, alpha, beta, gamma are parameters for ARIMA.
    # Upper case vars represent vectors.
    # sample_n is the number of samples.
    # Returns S the timeseries data matrix.
    np.random.seed(random_seed)

    MU = np.random.rand(sample_n,1)
    ALPLA =  np.random.rand(sample_n,1)
    BETA = np.random.rand(sample_n,1)
    SIGMA = 0.1 * np.random.rand(sample_n,1)

    S = np.zeros((MU.shape[0], p*T)) ### S0=1

    for i in range(MU.shape[0]):
        mu = MU[i]
        alpha = ALPLA[i]
        beta = BETA[i]
        sigma = SIGMA[i]
        w = np.random.randn(T) * sigma
        s = np.zeros(T)

        for t in range(len(w)):
            s[t] = mu + alpha * s[t-1] + w[t] + beta * w[t-1]
        
        S[i, :]=s

    return S, MU, ALPLA, BETA, SIGMA


def Periodic_gen(random_seed = 326, sample_n = sample_n):    

    np.random.seed(random_seed)

    W = np.random.randint(low=3, high=6, size=(sample_n, 1))
    H = np.random.randint(low=10, high=20, size=(sample_n, 1))
    L = np.random.randint(low=2, high=7, size=(sample_n, 1)) 
    S = np.zeros((sample_n, T))

    for ii in range(sample_n):
        for tt in range(T):
            if tt%24 >= 12 - int(W[ii, :]/2) and tt%24 <= 12 + int(W[ii, :]/2):
                S[ii, tt] = H[ii, :] + 0.3 * np.random.randn()
            else:
                S[ii, tt] = L[ii, :] + 0.3 * np.random.randn()

    # LineChart(x=[np.arange(len(S[-1]))], y=[S[-1]], legend=["S"], plot_show=True)
    return S, W, H, L

if __name__ == '__main__':
    # Phi = CtrlTask()
    # print(Phi.shape)

    # S, MU, ALPLA, BETA, SIGMA = ARIMA_gen()
    # print(S.shape, S)
    # S, MU, ALPLA, BETA, SIGMA = ARIMA_gen(random_seed=327)
    # print(S.shape, S)

    Periodic_gen()
