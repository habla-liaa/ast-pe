from paips.core import Task
import numpy as np
import pandas as pd
from IPython import embed

class MixupDP(Task):
    def process(self):
        data = self.parameters['in']
        col_audio = self.parameters['column_audio']
        col_label = self.parameters['column_label']
        alpha = self.parameters['alpha']
        prob = self.parameters['prob']
        original_idx = data.index
        data_mix = data.sample(len(data)).reset_index()
        alphas = np.random.beta(alpha,alpha, size=(len(data),))
        gate = np.random.uniform(low=0,high=1,size=(len(data,)))
        alphas = alphas*(gate<prob).astype(np.float32) #If deactivated, alpha = 0 and only data survives (original data)
        data[col_audio] = alphas*data_mix[col_audio].values + (1-alphas)*data[col_audio].values
        data[col_label] = alphas*data_mix[col_label].values + (1-alphas)*data[col_label].values
        return data

class RollDP(Task):
    def process(self):
        data = self.parameters['in']
        roll_range = self.parameters['roll_range']
        roll_prob = self.parameters['prob']
        roll_axis = self.parameters.get('roll_axis',-1)
        col_in = self.parameters['column_in']
        col_out = self.parameters['column_out']

        def fn(x):
            roll_dice = np.random.uniform(low=0,high=1)
            if roll_dice < roll_prob:
                roll_n = np.random.randint(low=roll_range[0],high=roll_range[1])
                return np.roll(x, roll_n, axis=roll_axis)
            else:
                return x

        data[col_out] = data[col_in].apply(fn)
        return data