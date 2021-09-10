from IPython.terminal.embed import embed
from tensorflow.keras.utils import Sequence
from paips.core import Task, TaskIO

import copy
import ray
import numpy as np
import pandas as pd

class KerasGenerator(Sequence):
    def __init__(self, 
                 data=None, 
                 data_generating_task=None, 
                 sync_first_iteration=True,
                 initial_iteration=0,
                 data_processing_task=None,
                 data_processing_mods=None,
                 batch_size=64,
                 x=None,
                 y=None,
                 steps_per_epoch=None,
                 shuffle=True,
                 extra_data=None):

        self._began_training = False
        self._generating_task_iteration = initial_iteration
        self._sync_first_iteration = sync_first_iteration
        self._data_generating_task = data_generating_task
        self._data_processing_task = data_processing_task
        self._next_data_generating_iteration_step = 0
        self._temp_next_generation_step = None
        self.data = data
        self.batch_size = batch_size
        self.batch_x = x
        self.batch_y = y
        self.step = 0
        self.intraepoch_step = 0
        self.length = steps_per_epoch
        self.shuffle = shuffle
        
        if self.data is None and self._data_generating_task is None:
            raise Exception('If data is not provided, data_generating_task should be given')
        if self.data is None:
            self.data = {}
        else:
            self._index = np.array(self.data.index)
            if self.shuffle:
                self._index = np.random.permutation(self._index)
            if self.length is None:
                self.length = len(self.data)//self.batch_size + 1
            self._data_generating_task = None
        
        if data_processing_mods is not None:
            for mod in data_processing_mods:
                k = list(mod.keys())[0]
                v = mod[k]
                k = k.replace('.','/')
                self._data_processing_task.parameters[k] = v
                if k.startswith('Tasks'):
                    k_parts = k.split('/')
                    task_name = k_parts[1]
                    task = [t for t in self._data_processing_task._dependency_order if t.name == task_name][0]
                    task.parameters['/'.join(k_parts[2:])] = v
                    task.initial_parameters['/'.join(k_parts[2:])] = v

        if extra_data is not None and self._data_processing_task is not None:
            for k,v in extra_data.items():
                self._data_processing_task.parameters['in'][k] = TaskIO(v,'abcd',iotype='data',name='gen_extra',position=0)

    def set_initial_intraepoch(self,intraepoch, step, intraepoch_step, next_generation_step):
        self._generating_task_iteration = intraepoch
        self.step = step
        self.intraepoch_step = intraepoch_step
        self._temp_next_generation_step = next_generation_step

    def _get_outputs(self, outs, task_name, sync=False):
        if not sync:
            ray_data = outs['{}->ray_reference'.format(task_name)].load()
            output_names = outs['{}->output_names'.format(task_name)].load()
            if isinstance(ray_data,list) and len(ray_data) == 1:
                outs = ray_data[0]
            if isinstance(output_names,list):
                output_names = output_names[0]
            task_out = ray.get(outs)
            outs = {'{}->{}'.format(task_name, out_name): val 
            for val, out_name in zip(task_out, output_names)}
   
        for k,v in outs.items():
            if isinstance(v,TaskIO):
                out_data = v.load()
            else:
                out_data = v
            if isinstance(out_data,list) and len(out_data) == 1:
                out_data = out_data[0]

        return out_data

    def _assign_to_data(self, data, sync=False):
        task_name = self._data_generating_task.name
        data = self._get_outputs(data,task_name,sync=sync)
        if isinstance(data,pd.core.frame.DataFrame):
            self.data = data
        elif isinstance(data, dict):
            for k,v in data.items():
                if isinstance(data,ray._raylet.ObjectRef):
                    self.data[k.split('->')[-1]] = ray.get(data)
                else:
                    self.data[k.split('->')[-1]] = data
        elif isinstance(data, ray._raylet.ObjectRef):
            self.data = ray.get(data)
        else:
            raise Exception('Data returned by the data_generation_task is in a non recognized format')

    def _run_data_generating_task(self,iteration,sync=False, mods=None):
        task = copy.deepcopy(self._data_generating_task)
        if mods is None:
            mods = {}
        if sync:
            mods.update({'async': False})
        else:
            mods.update({'async': True})
        task.parameters.update(mods)
        outs = task.run(iteration = iteration)
        
        return outs

    def on_train_begin(self):
        if self._data_generating_task is not None:
            outs = self._run_data_generating_task(self._generating_task_iteration,sync=self._sync_first_iteration)
            self._generating_task_outputs_to_be_assigned = outs
            self._began_training = True

    def _on_data_exhaust(self):
        #print('On data exhaust')
        self._generating_task_iteration += 1
        #print('Assigning data {}'.format(type(self._generating_task_outputs_to_be_assigned)))
        self._assign_to_data(self._generating_task_outputs_to_be_assigned, sync = self._generating_task_iteration == 1 or len(self.data) == 0)
        
        if self._temp_next_generation_step is not None:
            self._next_data_generating_iteration_step = self._temp_next_generation_step
            self._temp_next_generation_step = None
        else:
            self._next_data_generating_iteration_step = self.step + len(self.data)//self.batch_size

        self._index = list(self.data.index)
        self.intraepoch_step = 0

        #print('Launching new task')
        outs = self._run_data_generating_task(self._generating_task_iteration,sync=False)
        
        self._generating_task_outputs_to_be_assigned = outs

    def get_intra_epoch(self):
        return self._generating_task_iteration - 1

    def get_step_in_intra_epoch(self):
        return len(self._index)//self.batch_size - (self._next_data_generating_iteration_step - self.step)

    def get_next_intra_epoch_step(self):
        return self._next_data_generating_iteration_step

    def _get_batch(self,step):
        #if self.data['partition'].unique()[0] != 'train':
        #    from IPython import embed
        #    embed()
        batch_idxs = np.take(self._index,np.arange(step*self.batch_size,(step+1)*self.batch_size),mode='wrap',axis=0)
        batch_data = self.data.loc[batch_idxs]
        if self._data_processing_task is not None:
            batch_data = TaskIO(batch_data,'abcd',iotype='data',name='batch_i',position=0)
            self._data_processing_task.parameters['in']['batch_data'] = batch_data

            outs = self._data_processing_task.run()
            outs = {k.split('->')[-1]: v for k,v in outs.items()}

            if not isinstance(self.batch_x,list):
                self.batch_x = [self.batch_x]
            x = [outs[k].load() for k in self.batch_x]
            x = x[0] if len(x) == 1 else x

            if self.batch_y is not None:
                if not isinstance(self.batch_y,list):
                    self.batch_y = [self.batch_y]   

                y = [outs[k].load() for k in self.batch_y]
                y = y[0] if len(y) == 1 else y
            else:
                y = None
        else:
            x = [self.data[k].values for k in self.batch_x]
            y = [self.data[k].values for k in self.batch_y]

        return x,y

    def __getitem__(self,step):
        if self._began_training == False:
            self.on_train_begin()
        if self._data_generating_task is not None and self.step == self._next_data_generating_iteration_step or self._data_generating_task is not None and len(self.data) == 0:
            self._on_data_exhaust()
        
        if self._data_generating_task is not None:
            x,y = self._get_batch(self.intraepoch_step)
        else:
            x,y = self._get_batch(step)

        self.step += 1
        self.intraepoch_step += 1

        return x,y

    def __len__(self):
        return self.length

    def on_epoch_end(self):
        if self.shuffle and self.data is not None:
            self._index = np.random.permutation(self._index)
            print('Shuffled data')

class BatchGenerator2(Task):
    def process(self):
        pruned_parameters = {k: v for k,v in self.parameters.items() if k not in ['class','in_memory','cache','iter']}
        return KerasGenerator(**pruned_parameters)

