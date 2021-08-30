from paips.core import Task
import numpy as np
import pandas as pd
import copy
import soundfile as sf
import h5py
import librosa

class DownsampleDP(Task):
    def process(self):
        axis = self.parameters.get('axis',0)
        col_in = self.parameters['column_in']
        col_out = self.parameters.get('column_out',col_in)
        factor = self.parameters['factor']
        mode = self.parameters.get('mode','mean')

        data = self.parameters['in']

        def fn(x):
            target_axis_dim = x.shape[axis] + factor-(x.shape[axis]%factor)
            original_shape = list(x.shape)
            target_shape = copy.deepcopy(original_shape)
            target_shape[axis] = target_axis_dim

            y = np.zeros(target_shape)

            slc = [slice(None)] * len(target_shape)
            slc[axis] = slice(0, x.shape[axis])
            y[slc] = x
            y = np.swapaxes(y,axis,-1)
            reshape_shape = list(y.shape)
            reshape_shape[-1] = reshape_shape[-1]//factor
            reshape_shape.append(factor)
            y = np.reshape(y,reshape_shape)
            if mode == 'mean':
                y = np.mean(y,axis=-1)
            y = np.swapaxes(y,axis,-1)

            return y
            
        
        data[col_out] = data[col_in].apply(fn)
        return data

class LoadDataframeDP(Task):
    def process(self):
        col_in = self.parameters['column_in']
        col_out = self.parameters.get('column_out',col_in)
        data = self.parameters['in']
        exclude_cols = self.parameters.get('exclude_cols',None)
        include_cols = self.parameters.get('include_cols',None)
        order_by = self.parameters.get('order_by',None)
        
        def fn(x):
            if type(x).__name__ == 'GenericFile':
                x = x.load()
            if order_by is not None:
                x = x.sort_values(by = order_by)
            if exclude_cols is not None:
                x = x.drop(exclude_cols,axis=1)
            if include_cols is not None:
                x = x[include_cols]
            
            original_cols = list(x.columns)

            x = x.apply(lambda x: x.values,axis=1)
            x = np.stack(x.values)
            return x, original_cols

        y = list(zip(*map(fn,data[col_in])))
        out1 = pd.Series(y[0])
        out1.index = data.index
        out2 = pd.Series(y[1])
        out2.index = data.index
        data[col_out] = out1

        original_cols = out2.iloc[0]

        self.output_names = ['out', 'columns']
        return data, original_cols

class LSDecompositionDP(Task):
    def medianfilter(self,array,L,S):
        nrows = ((array.size-L)//S+1)
        n = array.strides[0]
        strided = np.lib.stride_tricks.as_strided(array, shape = (nrows,L), strides = (S*n,n))
        medianfiltered = np.median(strided,axis=1)
        return medianfiltered

    def process(self):
        data = self.parameters['in']
        col_in = self.parameters['column_in']
        col_out = self.parameters['column_out']
        filter_size = self.parameters.get('filter_size',80)
        def long_short_decomposition(melspec):
            melspec_ls = np.array([self.medianfilter(freq,filter_size,1) for freq in melspec.T])
            melspec_ss = melspec[:melspec_ls.shape[-1],:].T - melspec_ls
            return melspec_ls.T, melspec_ss.T

        y = list(zip(*map(long_short_decomposition,data[col_in])))
        ls = pd.Series(y[0])
        ls.index = data.index
        ss = pd.Series(y[1])
        ss.index = data.index
        data[col_out[0]] = ls
        data[col_out[1]] = ss

        return data

class MelspectrogramDP(Task):
    def process(self):
        data = self.parameters.pop('in')
        column_in, column_out = self.parameters.pop('column_in'), self.parameters.pop('column_out')
        pop_parameters = ['class','cache','in_memory']
        for p in pop_parameters:
            if p in self.parameters:
                self.parameters.pop(p)
        if 'log' in self.parameters:
            log = self.parameters.pop('log')
        else:
            log = True
        def fn(x):
            melspec = librosa.feature.melspectrogram(x, **self.parameters)
            if log:
                melspec = np.log(melspec + 1e-12)
            return melspec.T

        data[column_out] = data[column_in].apply(fn)
        return data

class NormalizeDP(Task):
    def process(self):
        data = self.parameters['in']
        statistics = self.parameters.get('statistics')
        column_in = self.parameters['column_in']
        column_out = self.parameters['column_out']

        if not isinstance(column_in, list):
            column_in = [column_in]
        if not isinstance(column_out, list):
            column_out = [column_out]

        columns = self.parameters.get('columns',None)
        if statistics is None:
            for col_in, col_out in zip(column_in,column_out):
                data[col_out] = data[col_in].apply(lambda x: (x - np.mean(x))/np.std(x))
        else:
            if len(statistics.keys()) > 0:
                for col_in, col_out in zip(column_in,column_out):
                    col_stats = statistics[col_in]
                    group = list(col_stats.keys())[0]
                    if group == 'global':
                        g_stats = col_stats[group]
                        if type(g_stats['mean']).__name__ == 'Series':      
                            if columns is not None:
                                g_stats['mean'] = g_stats['mean'].loc[columns]
                                g_stats['mean'] = g_stats['mean'].values
                        if type(g_stats['std']).__name__ == 'Series':
                            if columns is not None:
                                g_stats['std'] = g_stats['std'].loc[columns]
                            g_stats['std'] = g_stats['std'].values
                        data[col_out] = data.apply(lambda x: (x[col_in] - g_stats['mean'])/g_stats['std'],axis=1)
                    else:
                        for g, g_stats in col_stats[group].items():
                            if type(g_stats['mean']).__name__ == 'Series':
                                if columns is not None:
                                    g_stats['mean'] = g_stats['mean'].loc[columns]
                                g_stats['mean'] = g_stats['mean'].values
                            if type(g_stats['std']).__name__ == 'Series':
                                if columns is not None:
                                    g_stats['std'] = g_stats['std'].loc[columns]
                                g_stats['std'] = g_stats['std'].values

                        data[col_out] = data.apply(lambda x: (x[col_in] - col_stats[group][x[group]]['mean'])/(col_stats[group][x[group]]['std']),axis=1)
        return data

class OneHotVectorDP(Task):
    def process(self):
        data = self.parameters['in']
        column_in = self.parameters['column_in']
        column_out = self.parameters['column_out']
        mask = self.parameters.get('mask',None)
        frame_len = self.parameters.get('frame_len',None)
        n_classes = int(self.parameters['n_classes'])

        def fn(x, mask):
            if frame_len:
                hotvector = np.zeros((frame_len,n_classes))
                hotvector[:,x] = 1
            else:
                hotvector = np.zeros((n_classes))
                hotvector[x] = 1
            if mask is not None:
                slice_mask = [slice(None)] * 2 + [0]*(mask.ndim-2)
                mask = mask[slice_mask]

                last_idx = np.max(np.argwhere(np.all(mask == 1,axis=1)))
                mask = np.ones((len(mask),n_classes))
                if last_idx + 1 < len(mask):
                    mask[last_idx+1:] = 0
                return hotvector*mask
            else:
                return hotvector

        if mask is not None:
            data[column_out] = data.apply(lambda x: fn(x[column_in],x[mask]),axis=1)
        else:
            data[column_out] = data[column_in].apply(lambda x: fn(x,None))

        return data

class PadDP(Task):
    def process(self):
        data = self.parameters['in']
        col_in = self.parameters['column_in']
        col_out = self.parameters['column_out']
        max_length = self.parameters['max_length']
        axis = self.parameters.get('axis',0)
        
        def fn(x):
            if type(x).__name__ == 'GenericFile':
                x = x.load()
            shape_x = list(x.shape)
            shape_x[axis] = max_length
            out = np.zeros(shape_x)
            mask = np.zeros([max_length,])
            slc = [slice(None)] * len(x.shape)
            slc[axis] = slice(0, min(x.shape[axis],max_length))
            out[slc] = x[slc]
            mask[:x.shape[axis]] = 1
            return out, mask

        y = list(zip(*map(fn,data[col_in])))
        out1 = pd.Series(y[0])
        out2 = pd.Series(y[1])
        out1.index = data.index
        out2.index = data.index
        data[col_out] = out1
        data['mask'] = out2
        
        return data

class ReadAudioDP(Task):
    def process(self):
        column_end = self.parameters.get('column_end','end')
        column_filename = self.parameters.get('column_filename','filename')
        column_out = self.parameters.get('column_out','audio')
        column_sr = self.parameters.get('column_sr','sampling_rate')
        column_start = self.parameters.get('column_start','start')
        fixed_size = self.parameters.get('fixed_size',None)
        data_in = self.parameters.get('in',None)
        make_mono = self.parameters.get('make_mono',True)

        def extend_to_size(x,fixed_size):
            if fixed_size <= len(x):
                return x[:fixed_size]
            else:
                y = np.zeros((fixed_size,))
                y[:len(x)] = x
                return y

        def fn(x):
            y = sf.read(x[column_filename],start=int(x[column_start]),stop=int(x[column_end]))[0]

            if make_mono and y.ndim > 1:
                y = y[:,0]
            y = extend_to_size(y, fixed_size)
            return y

        data_in[column_out] = data_in.apply(fn,axis=1)

        return data_in

class ReadHDF5(Task):
    def pad_signal(self,x):
        if x.shape[0] < self.parameters['fixed_size']:
            pad_shape = list(x.shape)
            mask = np.zeros((self.parameters['fixed_size'],))
            mask[:pad_shape[0]] = 1
            pad_shape[0] = self.parameters['fixed_size'] - pad_shape[0]
            return np.concatenate([x,np.zeros(pad_shape)],axis=0), mask
        else:
            return x[:self.parameters['fixed_size']], np.ones((self.parameters['fixed_size'],))

    def process(self):
        if not hasattr(self, 'h5files'):
            self.h5files = {}
        data = self.parameters['in']
        if not isinstance(self.parameters['column_hdf5_key'],list):
            self.parameters['column_hdf5_key'] = [self.parameters['column_hdf5_key']]
        columns = self.parameters['column_hdf5_key']
        #If any h5py of the batch has still not be opened, then open it and store the pointer to the file
        for col in columns:
            for f in data['{}_h5file'.format(col)].unique():
                if f not in self.h5files:
                    self.h5files[f] = h5py.File(f, 'r')
            data[col] = data.apply(lambda row: self.h5files[row['{}_h5file'.format(col)]][row[col]][:],axis=1)
            if self.parameters.get('fixed_size',None):
                y = list(zip(*map(lambda x: self.pad_signal(x),data[col])))
                data[col] = pd.Series(y[0], index = data.index)
                data['padding_mask'] = pd.Series(y[1],index=data.index)

        return data

class SqueezeDP(Task):
    def process(self):
        data = self.parameters['in']
        col_in = self.parameters['column_in']
        col_out = self.parameters['column_out']

        data[col_out] = data[col_in].apply(lambda x: np.squeeze(x))

        return data

class ToNumpyDP(Task):
    def process(self):
        data = self.parameters['in']
        col_in = self.parameters['column_in']

        return np.stack(data[col_in])

class Wav2Vec2MaskDP(Task):
    def process(self):
        mask_shape = self.parameters.get('mask_shape',None)
        keep_value = self.parameters.get('keep_value',-1)
        mask_value = self.parameters.get('mask_value',0)
        consecutive_frames = self.parameters.get('consecutive_frames',10)
        column_in = self.parameters.get('column_in',None)
        column_out = self.parameters.get('column_out',None)
        p_mask = self.parameters.get('p_mask',0.065)
        p_same = self.parameters.get('p_same',0.2)
        data_in = self.parameters.get('in',None)

        def fn(x):
            if mask_shape is None:
                mask_loss = np.zeros_like(x)
            else:
                mask_loss = np.zeros(shape=mask_shape)
            p_mask_ = consecutive_frames*p_mask/len(mask_loss)
            p_same_ = p_same*p_mask_ #Probability that something masked is unmasked
            mask_same_starts = np.random.uniform(low=0,high=1,size=(len(mask_loss),))<p_same_
            mask_starts = np.random.uniform(low=0,high=1,size=(len(mask_loss),))
            mask_starts = np.logical_and(mask_starts>p_same_, mask_starts<p_mask_)
            start_idxs = np.argwhere(mask_starts).flatten()
            same_start_idxs = np.argwhere(mask_same_starts).flatten()

            mask_start_idxs = np.repeat(np.arange(consecutive_frames),len(start_idxs)) + np.tile(start_idxs,consecutive_frames)
            mask_same_idxs = np.repeat(np.arange(consecutive_frames),len(same_start_idxs)) + np.tile(same_start_idxs,consecutive_frames)

            mask_loss[mask_start_idxs[mask_start_idxs<len(mask_loss)]]=1
            mask_loss[mask_same_idxs[mask_same_idxs<len(mask_loss)]]=1

            mask_w2v = np.ones(shape=mask_shape)*keep_value
            mask_w2v[mask_start_idxs[mask_start_idxs<len(mask_loss)]]=0

            return mask_loss, mask_w2v
       
        temp_out = data_in[column_in].apply(fn)
        data_in['mask_loss'] = temp_out.apply(lambda x: x[0])
        data_in['mask_w2v'] = temp_out.apply(lambda x: x[1])

        return data_in