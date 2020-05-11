import numpy as np
weight1 = np.array([  1.0, -1.00000000e-02,  1.00000000e-02,  1.00000000e-02, -1.00000000e-02,
                        1.00000000e-02, -1.00000000e-02,  1.00000000e-02, -1.00000000e-02,
                        1.00000000e-02, -1.00000000e-02,  1.91267349e-05,  2.35327756e-05,
                        8.38857304e-05,  5.89284347e-06, -9.81139448e-05, -3.72979078e-07,
                       -1.49482813e-05,  3.68559351e-05, -5.81073866e-05, -2.29789573e-05,
                        6.80429634e-05,  2.87011047e-05,  1.33489562e-05,  8.94476256e-05,
                       -7.26650946e-05, -7.20767855e-05,  9.29151082e-05,  5.38008782e-05,
                       -7.23894306e-05,  9.74357590e-05, -4.93390542e-05,  8.00716766e-05,
                       -6.58275212e-05, -1.00169656e-05,  4.71867726e-05, -8.21092096e-05,
                        2.32290312e-06,  8.73113544e-05, -6.70825810e-05,  2.69354965e-05,
                       -2.44401009e-05, -7.91955701e-05, -1.29508629e-06,  3.02671462e-05,
                       -7.31276163e-05, -9.16792082e-05,  7.43416577e-06, -8.03227188e-05,
                       -7.42548696e-05,  5.20547349e-05, -6.09442379e-05, -6.20967326e-06,
                       -2.92131588e-05, -3.70642177e-05, -5.82352580e-05,  2.12492968e-05,
                       -8.57327699e-05, -6.81437481e-05,  9.60755793e-05,  3.71246042e-05,
                       -1.30656300e-05, -7.33830917e-05,  2.30589632e-05, -2.59987155e-05,
                        2.22335018e-05,  9.30952529e-05,  4.62012143e-05,  8.34012090e-05,
                       -9.04402730e-05, -8.90337378e-05, -3.81281374e-05,  1.00000000e-02,
                        2.00000000e-02,  1.00000000e-02,  1.00000000e-02, -8.29491434e-05,
                       -8.33015021e-05,  1.51715781e-06, -5.91228827e-05, -2.84881374e-05,
                        3.51241563e-05,  7.60154149e-05,  4.14462037e-05,  4.90807164e-05,
                        1.89865862e-05, -1.29809520e-05, -1.58316956e-05, -9.18427515e-05,
                        8.41876117e-05, -5.96186615e-05,  2.18577214e-05,  4.30630536e-05,
                       -1.78055570e-05, -1.88465220e-05,  4.19542985e-05, -8.89130789e-05,
                       -3.18735859e-05, -3.19322469e-05, -8.16860639e-05,  7.92077237e-05,
                        3.54249673e-05, -1.08255495e-06,  6.57760283e-05,  1.67837821e-05,
                       -8.03355418e-05,  5.45037866e-05,  4.05384486e-06,  1.22182098e-05,
                       -5.65322090e-05, -5.29005821e-06,  7.93921664e-05,  1.45037392e-06,
                        5.80785808e-06, -8.19769654e-05,  4.16720691e-05, -9.01850506e-06,
                       -9.40907707e-05, -9.07577102e-05,  7.02258277e-05, -9.75425196e-06,
                       -2.87265436e-05,  3.17469032e-05, -7.60074516e-05,  2.56751946e-05,
                       -4.22428111e-05, -5.18482641e-05,  5.99693638e-05,  3.49234354e-05,
                        3.11942618e-05,  2.24480669e-05, -7.41590957e-05,  7.17899761e-05,
                        2.75755242e-05,  3.95291432e-05,  6.74196082e-05, -2.63752151e-05,
                        3.64357792e-05, -2.92758282e-05,  4.35598964e-05, -9.45877121e-05,
                       -9.61504603e-05,  2.68919168e-05,  9.97031505e-05, -4.17068014e-05,
                        8.39527516e-05, -5.00659105e-05, -4.04473862e-05, -5.25981685e-05,
                        8.66869372e-05,  4.43737827e-05,  2.79078440e-05,  5.58548700e-05,
                        7.15582138e-05,  7.00736197e-05, -5.64579991e-05, -7.78304868e-05,
                        9.82069438e-05,  7.92260789e-05,  7.53220226e-05, -8.81103864e-05,
                       -6.17015481e-05,  2.87314349e-05,  2.76177564e-06, -4.49053680e-05,
                        5.64906884e-05, -3.87429872e-05, -5.21940874e-05,  2.57787311e-05,
                        2.79618235e-05,  8.83934498e-05,  9.94810155e-05,  3.10943045e-05,
                        5.59675927e-06,  9.00119780e-05,  4.67705841e-05, -9.91731886e-05,
                        8.50811735e-05, -7.65224550e-05, -8.08942809e-05,  5.16335934e-05,
                        7.73096270e-05, -5.39802846e-05,  7.53523409e-05,  8.51033278e-05,
                        9.78149157e-06,  7.66260431e-05, -6.97874364e-05, -2.31972583e-05,
                       -8.67089018e-05, -7.36132795e-05,  8.64128424e-05,  9.56944464e-07,
                       -5.05216105e-05,  8.83669065e-05, -5.42019535e-05,  1.80795937e-05,
                        8.33541578e-06, -3.36473782e-05,  4.93150405e-05, -2.58874487e-05,
                       -8.66154908e-05, -6.09489151e-05,  2.70924980e-05,  8.86562862e-05,
                       -5.01273132e-05, -8.74403128e-05, -9.31381562e-05, -6.99290776e-05,
                       -6.56742655e-05, -2.92766806e-05, -2.41965330e-05,  8.50798768e-05,
                       -1.96453451e-05, -3.36610582e-05, -7.44779030e-05,  7.39044190e-06,
                        3.52407515e-05, -4.53112284e-05,  4.51475241e-05,  4.46755061e-05,
                        5.89552756e-05,  6.96639773e-05,  3.88546816e-05, -5.14372086e-05,
                       -4.07771642e-05, -8.59578771e-05,  4.70192346e-05, -8.91343873e-05,
                        4.86261320e-05,  1.36053094e-05, -5.54511803e-05,  7.19140514e-05,
                        1.51637921e-05,  7.36146112e-05, -4.12674029e-05, -7.60788259e-05,
                       -4.39568155e-05, -9.13083421e-06,  6.83360369e-08, -6.38239406e-05,
                        5.68441888e-05, -2.15797972e-05,  8.07469360e-05,  1.26646140e-05,
                       -5.86964902e-05,  7.48302365e-05, -4.58366847e-05,  1.83582171e-05,
                       -3.62333392e-05,  5.90251094e-05, -3.06483116e-06, -3.84526981e-05,
                        4.06368798e-06, -7.66799257e-06,  1.62813049e-05, -7.88456131e-05,
                       -5.22800617e-05,  8.81325840e-05,  1.12725465e-05, -4.76347348e-05,
                       -8.71726005e-05,  4.76271169e-05,  6.05273405e-05, -2.73607773e-05,
                        2.63611934e-05, -3.96691892e-05, -5.34568474e-05,  4.38500186e-05,
                       -6.56845454e-05,  8.42991940e-05,  7.69897388e-05,  1.06474280e-05,
                        8.98198827e-05,  7.51360985e-05, -1.05003836e-05,  2.69106941e-05,
                       -2.41658595e-05,  1.53108743e-05, -2.70509402e-05,  2.12736072e-05,
                        8.85894641e-05, -7.19972557e-06, -8.07201506e-05,  6.61357920e-05,
                        5.14101433e-05,  8.09599949e-05,  2.11405306e-05,  1.30603476e-05,
                        6.50871256e-06, -3.76941062e-06,  6.92522143e-05,  6.95816237e-05,
                       -1.65984952e-05,  7.57432530e-05, -9.44762218e-05,  3.39578077e-05,
                        1.65714756e-05, -7.83175869e-05,  7.00180187e-05,  5.17558049e-05,
                       -1.88150895e-06,  5.98522977e-05,  6.03498099e-05, -6.26551809e-05,
                        2.21521390e-05,  7.90292580e-05,  1.61668213e-05, -9.76981737e-05])
