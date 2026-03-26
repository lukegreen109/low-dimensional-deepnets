CHOICES = {
    'seed': list(range(42, 52))+[0],
    'm': ["allcnn", "convmixer", "fc",
          "vit", "wr-10-4-8", "wr-16-4-64",
          "fc-200-512-5", "fc-200-512-512-5"],
    'opt': ["adam", "sgd", "sgdn", "geodesic"],
    'corner': ["uniform", "normal", "subsample-2000", "subsample-200"],
    'lr': [0.001, 0.1, 0.25, 0.5, 0.0005, 0.0025, 0.00125,
           0.005],
    # 'bs': [100, 200, 500],
    'bs':[100],
    'aug': ['simple', 'none'],
    'wd': [0., 1.e-03, 1.e-05],
    'iseed': range(0, 3),
    'isinit': [False, True],
}

CDICT_M = {'allcnn': '#e41a1c',
            'convmixer': '#377eb8',
            'fc': '#4daf4a',
            'vit': '#984ea3',
            'wr-10-4-8': '#ff7f00',
            'wr-16-4-64': '#ffff33',
            'geodesic': '#000000'}

CDICT_OPT = {'adam': '#e41a1c',
            'sgd': '#377eb8',
            'sgdn': '#4daf4a',
            'geodesic': '#000000'}

# Consistent (soft) colors for batch selection methods.
# Use this across notebooks/figures to keep method colors stable.
CDICT_BSEL = {
    'Uniform':  '#66c2a5',
    'RhoLoss':  '#fc8d62',
    'DivBS':    '#8da0cb',
    'Bayesian': '#e78ac3',
    'geodesic': '#000000',
    'Full':     "#e1e86e",
    'GradNorm': "#c76c6c",
}
