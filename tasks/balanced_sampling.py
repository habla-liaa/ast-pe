from paips.core import Task

import joblib
from pathlib import Path
import numpy as np

from IPython import embed

class BalancedSamplingMixup(Task):
    def process(self):
        data = self.parameters['in']
        if self.parameters.get('enable',True):
            counts = joblib.load(Path(self.parameters.get('counts')).expanduser().absolute())
            counts = counts.sort_index()
            labels_column = self.parameters.get('labels_column')
            source_column = self.parameters.get('source_column','filename')
            out_column = self.parameters.get('out_column')
            counts = counts['class_count'].values
            data['rarity'] = data[labels_column].apply(lambda x: np.sum(x*1.0/counts))
            mixup_samples_idx = np.random.choice(data.index, size=len(data),replace=True,p=data['rarity'].values/data['rarity'].sum())
            mixup_df = data.loc[mixup_samples_idx]
            mixup_df.index = data.index
            data[out_column] = mixup_df[source_column]
            data['mixup_labels'] = mixup_df[labels_column]
            data[labels_column] = data[labels_column] + data['mixup_labels']
            alpha = self.parameters.get('alpha',0.1)
            rate = self.parameters.get('rate',0.5)
            lambdas = np.random.beta(alpha,alpha, size=(len(data),))
            gate = np.random.uniform(low=0,high=1,size=(len(data,)))
            lambdas = lambdas*(gate<rate).astype(np.float32)
            data['mixup_lambda'] = lambdas
            if not self.parameters.get('mix_labels',False):
                data[labels_column] = data[labels_column].apply(lambda x: np.minimum(x,1))
        else:
            data['mixup_lambda'] = np.zeros((len(data),))

        return data
