from paips.core import Task, TaskIO
import numpy as np
from tensorflow.keras.utils import Sequence
import copy
import ray
from pyknife.aws import S3File
from pyknife.file import CompressedFile
import copy
import pandas as pd

class BatchGenerator(Task):
    def process(self, initial_epoch=0):
        on_epoch_end_task = self.parameters.get('on_epoch_end_task',None)
        epoch_on_begin = self.parameters.get('epoch_end_at_start', False)
        on_begin_train_task = self.parameters.get('on_begin_train_task',None)
        begin_train_task_as_epoch = self.parameters.get('begin_train_task_as_epoch',False)
        on_begin_train_mods = self.parameters.get('on_begin_train_mods',None)
        use_epoch_as_iteration = self.parameters.get('use_epoch_as_iteration',False)
        data_source = self.parameters.get('data_source',None)

        batch_size = self.parameters.get('batch_size',1)
        batch_x = self.parameters.get('x',None)
        batch_y = self.parameters.get('y',None)
        shuffle = self.parameters.get('shuffle',True)
        seed = self.parameters.get('seed',1234)
        batch_task =  copy.deepcopy(self.parameters.get('batch_task',None))
        extra_data = self.parameters.get('extra_data',None)
        steps_per_epoch = self.parameters.get('steps_per_epoch',None)

        if extra_data is not None:
            for k,v in extra_data.items():
                self.batch_task.parameters['in'][k] = TaskIO(v,'abcd',iotype='data',name='gen_extra',position=0)

        class KerasGenerator(Sequence):
            def __init__(self_gen):
                np.random.seed(seed)
                self_gen.epoch = initial_epoch
                self_gen.data = {}
                self_gen.on_begin_train_task = on_begin_train_task
                self_gen.shuffle = shuffle
                self_gen.batch_task = batch_task
                self_gen.batch_x = batch_x
                self_gen.batch_y = batch_y
                self_gen.on_epoch_end_task = on_epoch_end_task
                self_gen.intra_epoch = 1
                self_gen.step_in_intra_epoch = 0
                self_gen.intra_epoch_len = 0
                self_gen.initial_intraepoch = 1

                if isinstance(data_source,pd.core.frame.DataFrame):
                    self_gen.train_data = data_source
                    self_gen.idxs = np.array(self_gen.train_data.index)

            def do_shuffle(self_gen):
                self_gen.idxs = np.random.permutation(self_gen.idxs)

            def set_initial_intraepoch(self_gen, intraepoch, intraepoch_step):
                self_gen.initial_intraepoch = intraepoch - 1
                self_gen.intra_epoch = intraepoch - 1
                self_gen.step_in_intra_epoch = intraepoch_step


            def on_intra_epoch_end(self_gen):
                if (self_gen.intra_epoch == self_gen.initial_intraepoch and epoch_on_begin and self_gen.on_begin_train_task) or (self_gen.intra_epoch == 0 and self_gen.on_begin_train_task):
                    self_gen.organize_outputs(self_gen.task_output,self_gen.on_begin_train_task)
                    
                elif self_gen.on_epoch_end_task is not None:
                    self_gen.organize_outputs(self_gen.task_output,on_epoch_end_task)
                    
                if not isinstance(self_gen.data,pd.core.frame.DataFrame) and not isinstance(data_source,pd.core.frame.DataFrame):
                    self_gen.train_data = self_gen.data[data_source]

                if 'sync' in self.parameters:
                    print('Syncing files from workers')
                    for sync_config in self.parameters['sync']:
                        self_gen.sync(sync_config)

                self_gen.idxs = self_gen.train_data.index
                self_gen.intra_epoch_len = len(self_gen.idxs)//batch_size
                if self_gen.shuffle:
                    self_gen.do_shuffle()

                #Get next epoch data:
                if self_gen.on_epoch_end_task:
                    try:
                        if use_epoch_as_iteration:
                            outs = self_gen.execute_task(on_epoch_end_task,iteration=self_gen.intra_epoch)
                            self_gen.task_output = outs
                        else:
                            outs = self_gen.execute_task(on_epoch_end_task)
                            self_gen.task_output = outs
                    except Exception as e:
                        print('Could not run epoch end task: {}'.format(e))

                self_gen.intra_epoch += 1
                self_gen.step_in_intra_epoch = 1
                print('New intra epoch')

            def sync(self_gen, sync_config):
                if sync_config['source'].startswith('s3:'):
                    S3File(sync_config['source']).download(sync_config['destination'])
                else:
                    import shutil
                    destination_path = Path(sync_config['destination']).expanduser()
                    if not destination_path.parent.exists():
                        destination_path.parent.mkdir(parents=True)
                    shutil.copyfile(Path(sync_config['source']).expanduser(),destination_path)

                clean_extraction_path = sync_config.get('clean_extraction_path',False)
                if clean_extraction_path:
                    if Path(sync_config['extract_to']).expanduser().exists():
                        import shutil
                        shutil.rmtree(str(Path(sync_config['extract_to']).expanduser()))
                    
                if 'extract_to' in sync_config:
                    CompressedFile(sync_config['destination']).extract(sync_config['extract_to'])

            def execute_task(self_gen, task, iteration=None):
                outs = task.run(iteration = iteration)
                return outs

            def organize_outputs(self_gen,outs,task):
                if task.parameters.get('async',False):
                    ray_data = outs['{}->ray_reference'.format(task.name)].load()
                    output_names = outs['{}->output_names'.format(task.name)].load()
                    if isinstance(ray_data,list) and len(ray_data) == 1:
                        outs = ray_data[0]
                    if isinstance(output_names,list):
                        output_names = output_names[0]
                    task_out = ray.get(outs)
                    outs = {'{}->{}'.format(task.name, out_name): val 
                    for val, out_name in zip(task_out, output_names)}
   
                for k,v in outs.items():
                    if isinstance(v,TaskIO):
                        out_data = v.load()
                    else:
                        out_data = v
                    if isinstance(out_data,list) and len(out_data) == 1:
                        out_data = out_data[0]
                    if isinstance(out_data,ray._raylet.ObjectRef):
                        self_gen.data[k.split('->')[-1]] = ray.get(out_data)
                    else:
                        self_gen.data[k.split('->')[-1]] = out_data

            def on_train_begin(self_gen):
                if self_gen.on_begin_train_task:
                    if on_begin_train_mods:
                        if isinstance(on_begin_train_mods,dict):
                            self_gen.on_begin_train_task = copy.deepcopy(self_gen.on_begin_train_task)
                            self_gen.on_begin_train_task.parameters.update(on_begin_train_mods)
                    if begin_train_task_as_epoch:
                        outs = self_gen.execute_task(self_gen.on_begin_train_task,iteration=self_gen.initial_intraepoch)
                        self_gen.task_output = outs
                        self_gen.epoch += 1

                if epoch_on_begin:
                    self_gen.on_intra_epoch_end()

            def on_epoch_end(self_gen):
                self_gen.intra_epoch = 0

            def __getitem__(self_gen, idx):
                self_gen.step_in_intra_epoch += 1
                if self_gen.step_in_intra_epoch == self_gen.intra_epoch_len:
                    self_gen.on_intra_epoch_end()
                    
                batch_idxs = np.take(self_gen.idxs,np.arange(idx*batch_size,(idx+1)*batch_size),mode='wrap',axis=0)
                batch_data = self_gen.train_data.loc[batch_idxs]

                batch_data = TaskIO(batch_data,'abcd',iotype='data',name='batch_i',position=0)

                self_gen.batch_task.parameters['in']['batch_data'] = batch_data

                outs = self_gen.batch_task.run()
                outs = {k.split('->')[-1]: v for k,v in outs.items()}

                if not isinstance(self_gen.batch_x,list):
                    self_gen.batch_x = [self_gen.batch_x]
                if not isinstance(self_gen.batch_y,list):
                    self_gen.batch_y = [self_gen.batch_y]                

                x = [outs[k].load() for k in self_gen.batch_x]
                y = [outs[k].load() for k in self_gen.batch_y]

                x = x[0] if len(x) == 1 else x
                y = y[0] if len(y) == 1 else y

                return x,y
                
            def __len__(self_gen):
                # Ver como hacer para que cambie en cada epoca
                if steps_per_epoch is not None:
                    return steps_per_epoch
                else:
                    return len(self_gen.train_data)//batch_size + 1
        
        return KerasGenerator()