import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

nass = pd.read_csv('/home/thomas/mt/statistics/nass_merged.csv').dropna()

nass = nass[nass['ST_CNTY_STR'].str.startswith('MT')]

good = ['ST_CNTY_STR', 'VALUE_2002', 'VALUE_2007', 'VALUE_2012', 'VALUE_2017']
nass = nass[good].T
nass.columns = [s.replace('MT_', '').lower().replace(' ', "_") for s in nass.loc['ST_CNTY_STR']]

nass = nass.T.drop('ST_CNTY_STR', axis=1)
unet = pd.read_csv('/home/thomas/mt/statistics/irrigated_acreage_cnn_sept28.csv')

years = [2002,2007,2012,2017]
unet = unet.T
unet.columns = [int(a) for a in unet.loc['Unnamed: 0']]
unet = unet[years]
unet = unet.drop('Unnamed: 0', axis=0)
nass.columns = [int(p.replace('VALUE_', '')) for p in nass.columns]


# unet_total = unet.sum(axis=0)
# nass_total = nass.sum(axis=0)
# 
# plt.title('unet vs nass')
# plt.plot(unet_total, 'r-.')
# plt.plot(unet_total, 'rx', label='unet')
# plt.plot(nass_total, 'k-.')
# plt.plot(nass_total, 'kx', label='nass')
# plt.xlabel('year', fontsize=14)
# plt.ylabel('area', fontsize=14)
# plt.legend()
# plt.show()

from sklearn import linear_model

def get_correlations(a, b):
    coeff_det = r2_score(a, b)
    regr = linear_model.LinearRegression()
    regr.fit(a, b)
    pred = regr.predict(a)
    slope, intercept = regr.coef_[0][0], regr.intercept_
    return coeff_det, slope, intercept

nass = nass.loc[unet.index]
slog = False

import numpy as np

fig, ax = plt.subplots(ncols=2, nrows=4, sharex=True)

s = [0, 2e5]
n_bags = 100000
sz = 1.0
btstrp = True

slps = []
for i, year in enumerate(years):

    unet_ = np.asarray(unet[year].values).astype(np.float32).reshape(-1,1)
    nass_ = np.asarray(nass[year].values).astype(np.float32).reshape(-1,1)

    coeff_det, slope, intercept = get_correlations(nass_, unet_)

    if btstrp:
        slope_ = np.zeros((n_bags))
        r2 = np.zeros((n_bags))
        int_ = np.zeros((n_bags))
        for j in range(n_bags):
            indices = np.random.choice(unet.shape[0], replace=True, size=int(unet_.shape[0]*sz))
            coeff_det, slope, intercept = get_correlations(nass_[indices], unet_[indices])
            slope_[j] = slope
            r2[j] = coeff_det
            int_[j] = intercept

        slope = np.mean(slope_)
        slope_std = np.std(slope_)
        coeff_det = np.mean(r2)
        intercept = np.mean(int_)
        slps.append(slope_)


    ax[i, 0].plot(nass_, unet_, 'r.')
    ax[i, 0].set_xlim(s)
    ax[i, 0].set_ylim(s)
    ax[i, 0].plot(s, s, 'k--', label='1-1')
    ax[i, 0].plot(nass_, slope*nass_ + intercept, label='l.o.b.f')
    resid = unet_ - (slope*nass_ + intercept)
    ax[i, 1].plot(nass_, resid, 'k.')
    ax[i, 0].set_xlim(s)
    ax[i, 0].set_ylim(s)

    ax[i, 0].set_title(r'm: {:.3f}, m$_\sigma$: {:.3f}, r$^2$: {:.3f}'.format(slope, slope_std, coeff_det))

    ax[i, 1].set_title(r'resid. $\mu$: {:.3f}, $\sigma$: {:.3f}'.format(np.mean(resid),
        np.std(resid)))

ax[0,0].legend()



plt.suptitle("NASS vs UNET by county")
fig.text(0.5, 0.04, 'NASS irrigated acreage', ha='center', fontsize=14)
fig.text(0.04, 0.5, 'unet irrigated acreage', va='center', rotation='vertical', fontsize=14)
plt.show()

fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True)
for i, slp in enumerate(slps):
    ax[i // 2, i%2].hist(slp)
    ax[i // 2, i%2].set_title(slp.mean())
plt.show()
