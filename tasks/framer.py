from paips.core import Task
import tqdm
import pandas as pd
import numpy as np

class Framer(Task):
    def frame_row(self,row):
        try:
            logid, row = row
            time_column = self.parameters.get('time_column',None)
            if time_column is None:
                raise Exception('Missing time_column parameter')
            max_time = float(row[time_column])

            if max_time<=self.parameters['frame_size'] and self.parameters.get('window_short_audios',False):
                starts = [0]
                ends = [max_time]
            else:
                starts = np.arange(0,max_time-self.parameters['frame_size'],self.parameters['hop_size'],dtype=int)
                ends = starts + self.parameters['frame_size']

            new_ids = ['{}_{}'.format(logid,i) for i in range(len(starts))]
            row_dict = {'start': starts,
                        'end': ends,
                        'logid': new_ids,
                        'frame_parent': logid}
            for k,v in row.iteritems():
                if k not in row_dict:
                    if (isinstance(v,list) or isinstance(v,np.ndarray)) and len(v) != len(new_ids):
                        row_dict[k] = [v]*len(new_ids)
                    else:
                        row_dict[k] = v
        except:
            row_dict = {'start': [],
                        'end': [],
                        'logid': [],
                        'frame_parent': []}
                        
        return pd.DataFrame(row_dict).set_index('logid')

    def process(self):
        df_data = self.parameters['in']
        out_dfs = []

        if self.parameters.get('run_parallel',False):
            from ray.util.multiprocessing.pool import Pool
            import os
            def set_niceness(niceness): # pool initializer
                os.nice(niceness)
            n_cores = self.parameters.get('n_cores',4)
            pool = Pool(processes=n_cores,initializer=set_niceness,initargs=(20,),ray_address="auto") #(Run in same host it was called)
            out_dfs = pool.map(self.frame_row,df_data.iterrows(),chunksize=len(df_data)//len(pool._actor_pool))
        else:
            out_dfs = [self.frame_row(x) for x in df_data.iterrows()]

        out = pd.concat(out_dfs)
        return out
