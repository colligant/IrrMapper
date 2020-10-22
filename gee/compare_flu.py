import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

flu = pd.read_csv('/home/thomas/mt/statistics/irrigated_acreage_flu_oct21.csv').dropna()
flu = flu.T
flu.columns = [int(a) for a in flu.loc['Unnamed: 0']]
flu = flu.drop('Unnamed: 0', axis=0)

unet = pd.read_csv('/home/thomas/mt/statistics/irrigated_acreage_cnn_sept28.csv')

years = [2010, 2011, 2013, 2015, 2017, 2018, 2019]

unet = unet.T
unet.columns = [int(a) for a in unet.loc['Unnamed: 0']]
unet = unet[years]
unet = unet.drop('Unnamed: 0', axis=0)

flu.columns = unet.columns

unet_total = unet.sum(axis=0)
flu_total = flu.sum(axis=0)

# plt.title('unet vs flu')
# plt.plot(unet_total, 'r-.')
# plt.plot(unet_total, 'rx', label='unet')
# 
# plt.plot(flu_total, 'k-.')
# plt.plot(flu_total, 'kx', label='flu')
# plt.legend()
# plt.show()


flu = flu.loc[unet.index]

# 0, 0
# 0, 1
# 0, 2
# 0, 3
# 1, 0
# 1, 1
# 1, 2

from sklearn import linear_model
def get_correlations(a, b):
    coeff_det = r2_score(a, b)
    regr = linear_model.LinearRegression()
    regr.fit(a, b)
    pred = regr.predict(a)
    slope, intercept = regr.coef_[0][0], regr.intercept_
    return coeff_det, slope, intercept

import numpy as np
slog = False

fig, ax = plt.subplots(ncols=2, nrows=7, sharex=True)

s = [0, 2e5]
n_bags = 1000
sz = 1.0
btstrp = True

for i, year in enumerate(years):

    unet_ = np.asarray(unet[year].values).astype(np.float32).reshape(-1,1)
    flu_ = np.asarray(flu[year].values).astype(np.float32).reshape(-1,1)

    coeff_det, slope, intercept = get_correlations(flu_, unet_)

    if btstrp:
        slope_ = np.zeros((n_bags))
        r2 = np.zeros((n_bags))
        int_ = np.zeros((n_bags))
        for j in range(n_bags):
            indices = np.random.choice(unet.shape[0], replace=True, size=int(unet_.shape[0]*sz))
            coeff_det, slope, intercept = get_correlations(flu_[indices], unet_[indices])
            slope_[j] = slope
            r2[j] = coeff_det
            int_[j] = intercept

        slope = np.mean(slope_)
        coeff_det = np.mean(r2)
        intercept = np.mean(int_)

    ax[i, 0].plot(flu_, unet_, 'r.')
    ax[i, 0].set_xlim(s)
    ax[i, 0].set_ylim(s)
    ax[i, 0].plot(s, s, 'k--', label='1-1')
    ax[i, 0].plot(flu_, slope*flu_ + intercept, label='l.o.b.f')
    resid = unet_ - (slope*flu_ + intercept)
    ax[i, 1].plot(flu_, resid, 'k.')
    ax[i, 0].set_xlim(s)
    ax[i, 0].set_ylim(s)

    ax[i, 0].set_title(r'm: {:.3f}, r$^2$: {:.3f}'.format(slope, coeff_det))

    ax[i, 1].set_title(r'resid. $\mu$: {:.3f}, $\sigma$: {:.3f}'.format(np.mean(resid),
        np.std(resid)))

ax[0,0].legend()



plt.suptitle("FLU vs UNET by county")
fig.text(0.5, 0.04, 'FLU irrigated acreage', ha='center', fontsize=14)
fig.text(0.04, 0.5, 'unet irrigated acreage', va='center', rotation='vertical', fontsize=14)
plt.show()
