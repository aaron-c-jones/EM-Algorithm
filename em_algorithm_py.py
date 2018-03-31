"""
@author: aaronjones
@date: 2018-03-30
"""

import numpy

numpy.random.seed(0)

n = 500
mu = numpy.array([3., -2., 1.])
sig = numpy.array(
    [[10., 5., 4.], [5., 18., 7.], [4., 7., 9.]]
)
data = numpy.random.multivariate_normal(
    mean=mu, cov=sig, size=n
)

index = numpy.array([
    numpy.random.choice(a=[1, numpy.nan], size=n, p=[0.8, 0.2]),
    numpy.random.choice(a=[1, numpy.nan], size=n, p=[0.8, 0.2]),
    numpy.random.choice(a=[1, numpy.nan], size=n, p=[0.8, 0.2])
]).T

missing_data = numpy.multiply(data, index)


def em_algorithm(
    data, max_it=100, tol_err=1e-12
):
    data = data[~numpy.isnan(data).all(axis=1)]

    n, p = data.shape
    mod_rel_err = 1.0
    it = 1.0
    mu_init = numpy.nanmean(data, axis=0)

    predicted_df = data.copy()

    for i in range(p):
        predicted_df[numpy.isnan(data[:, i]), i] = mu_init[i]

    sig_init = (n - 1) / float(n) * numpy.cov(predicted_df.T)

    sig_init_reshape = sig_init.reshape(1, sig_init.size)
    theta_init = (
        numpy.append(mu_init, sig_init_reshape)
        .reshape(sig_init_reshape.size + mu_init.size, 1)
    )

    while mod_rel_err > tol_err and it <= max_it:
        temp_m = temp_s = 0

        for i in range(n):
            x_st = data[i, :].reshape(p, 1).copy()

            if numpy.isnan(x_st).sum() != 0:
                pos = numpy.argwhere(numpy.isnan(x_st))[:, 0]
                x_st[pos, :] = (
                    mu_init[pos].reshape(len(pos), 1)
                    + numpy.dot(
                        numpy.dot(
                            numpy.delete(sig_init[pos, :], pos, 1),
                            numpy.linalg.inv(
                                numpy.delete(
                                    numpy.delete(
                                        sig_init, pos, 0
                                    ), pos, 1
                                )
                            )
                        ),
                        numpy.delete(x_st, pos, 0)
                        - numpy.delete(mu_init.reshape(p, 1), pos, 0)
                    )
                )
                predicted_df[i, :] = x_st.T

            temp_m = temp_m + x_st

        mu_updated = temp_m / float(n)

        for i in range(n):
            x_st = data[i, :].reshape(p, 1)
            s_st = numpy.dot(x_st, x_st.T)
            predicted_df_temp = predicted_df[i, :]

            if numpy.isnan(x_st).sum() != 0:
                pos = numpy.argwhere(numpy.isnan(x_st))[:, 0]
                s_st[pos[:, None], pos] = (
                    sig_init[pos[:, None], pos]
                    - numpy.dot(
                        numpy.dot(
                            numpy.delete(sig_init[pos, :], pos, 1),
                            numpy.linalg.inv(
                                numpy.delete(
                                    numpy.delete(
                                        sig_init, pos, 0
                                    ), pos, 1
                                )
                            )
                        ),
                        numpy.delete(sig_init[:, pos], pos, 0)
                    )
                    + numpy.dot(
                        predicted_df_temp[pos].reshape(
                            len(predicted_df_temp[pos]), 1
                        ),
                        predicted_df_temp[pos].reshape(
                            1, len(predicted_df_temp[pos])
                        )
                    )
                )
                s_st[numpy.delete(numpy.arange(p), pos, 0)[:, None], pos] = (
                    numpy.dot(
                        numpy.delete(x_st, pos, 0),
                        predicted_df_temp[pos].reshape(
                            1, len(predicted_df_temp[pos])
                        )
                    )
                )
                s_st[pos, numpy.delete(numpy.arange(p), pos, 0)[None, :]] = (
                    s_st[numpy.delete(numpy.arange(p), pos, 0), pos]
                )

            temp_s = temp_s + s_st

        sig_updated = temp_s / float(n) - numpy.dot(mu_updated, mu_updated.T)

        sig_updated_reshape = sig_updated.reshape(1, sig_updated.size)
        theta_updated = (
            numpy.append(mu_updated, sig_updated_reshape)
            .reshape(sig_updated_reshape.size + mu_updated.size, 1)
        )

        mod_rel_err = (
            numpy.linalg.norm(theta_updated - theta_init)
            / max([1, numpy.linalg.norm(theta_init)]))

        print(it, mod_rel_err)

        it = it + 1

        mu_init = mu_updated.copy()
        sig_init = sig_updated.copy()
        theta_init = theta_updated.copy()

    return mu_updated, sig_updated, predicted_df


mu_em, sig_em, pred_em = em_algorithm(missing_data)

print('Original Mu: \n {0}'.format(numpy.round(mu, decimals=2)))
print('Imputed Mu: \n {0}'.format(numpy.round(mu_em, decimals=2)))

print('Original Sigma: \n {0}'.format(numpy.round(sig, decimals=2)))
print('Imputed Sigma: \n {0}'.format(numpy.round(sig_em, decimals=2)))

print('Missing Data: \n {0}'.format(numpy.round(missing_data, decimals=2)))
print('Imputed Data: \n {0}'.format(numpy.round(pred_em, decimals=2)))
