from paips.core import Task
import tqdm
import pandas as pd
import numpy as np

from dienen import Model
from kahnfigh import Config
import os
import copy

import glob
from pathlib import Path
import joblib

from timeit import default_timer as timer

from pyknife.aws import S3File
from paips.utils import GenericFile
import h5py

from IPython import embed

class DienenModel(Task):
    def process(self):
        if self.parameters.get('wandb_run',None):
            parents = ['/'.join(p.split('/')[:-1]) for p in self.parameters.find_path('wandb_run')]
            for parent in parents:
                parent = parent + '/wandb_run'
                self.parameters[parent] = self.parameters['wandb_run']
            import wandb
            wandb.init(id = self.parameters['wandb_run'], resume=self.parameters['wandb_run'])

        if 'epochs' in self.parameters:
            self.parameters['dienen_config/Model/Training/n_epochs'] = self.parameters['epochs']

        dienen_model = Model(self.parameters['dienen_config'])

        seed = self.parameters.get('seed',1234)
        dienen_model.set_seed(seed)

        dienen_model.set_model_path(self.cache_dir)

        cache_dir = GenericFile(self.cache_dir,'checkpoints')
        ckpt_exportdir = GenericFile(self.export_path,'checkpoints')

        if not Path(ckpt_exportdir.local_filename).exists():
            if not Path(cache_dir.local_filename).exists():
                cache_dir.mkdir(parents=True,exist_ok=True)
            if not ckpt_exportdir.parent.exists():
                ckpt_exportdir.parent.mkdir(parents=True,exist_ok=True)

            if not Path(ckpt_exportdir.local_filename).exists() and not Path(ckpt_exportdir.local_filename).is_symlink():
                os.symlink(cache_dir.local_filename,ckpt_exportdir.local_filename)
                
        train_data = self.parameters['train_data']
        validation_data = self.parameters.get('validation_data',None)

        dienen_model.set_data(train_data, validation = validation_data)

        keras_model = dienen_model.build()

        #Resume training if checkpoints exist
        last_epoch = -1
        last_step = 0
        last_intraepoch = None
        last_intraepoch_step = None
        finished_training = False
        
        if self.cache:
            metadata_path = GenericFile(cache_dir,'metadata')
            if metadata_path.exists():
                #Abrir metadata y buscar ultimo y mejor checkpoints
                metadata = metadata_path.load()
                last_ckpt = max([ckpt['step'] for ckpt in metadata])
                last_ckpt = [ckpt for ckpt in metadata if ckpt['step'] == last_ckpt][0]

                best_ckpt = [epoch['metric_val'] for epoch in metadata if 'metric_val' in epoch]
                if len(best_ckpt) > 0:
                    metric_name = metadata[-1]['metric']
                    if metric_name.endswith('loss'): #Revisar criterio, hay seguro mas casos en los que es mejor minimizar
                        best_ckpt = [ckpt for ckpt in metadata if ('metric_val' in ckpt) and (ckpt['metric_val'] == min(best_ckpt))]
                    else:
                        best_ckpt = [ckpt for ckpt in metadata if ('metric_val' in ckpt) and (ckpt['metric_val'] == max(best_ckpt))]
                    best_ckpt = best_ckpt[0]

                #Chequear si hay early stopping:
                has_earlystop = Config(self.parameters['dienen_config']).find_keys('EarlyStopping')

                if len(has_earlystop)>0:
                    patience = Config(self.parameters['dienen_config'])[has_earlystop[0]].get('patience',1)
                    patience_unit = Config(self.parameters['dienen_config'])[has_earlystop[0]].get('patience_unit','epoch')

                    if last_ckpt[patience_unit] - best_ckpt[patience_unit] >= patience:
                        last_epoch = self.parameters['dienen_config/Model/Training/n_epochs']
                        last_step = last_ckpt['step']
                        finished_training = True
                        weights_path = best_ckpt['weights_path']
                        opt_weights_path = best_ckpt['opt_weights_path']
                    else:
                        last_epoch, last_step, last_intraepoch, last_intraepoch_step, next_intra_epoch_step = last_ckpt['epoch'], last_ckpt['step'], last_ckpt['intra_epoch'], last_ckpt['intra_epoch_step'], last_ckpt['next_intra_epoch_step'] 
                        weights_path = best_ckpt['weights_path']
                        opt_weights_path = best_ckpt['opt_weights_path']
                else:
                    weights_path = last_ckpt['weights_path']
                    opt_weights_path = last_ckpt['opt_weights_path']
                    last_epoch, last_step, last_intraepoch, last_intraepoch_step, next_intra_epoch_step = last_ckpt['epoch'], last_ckpt['step'], last_ckpt['intra_epoch'], last_ckpt['intra_epoch_step'], last_ckpt['next_intra_epoch_step'] 

                self.cache_dir = GenericFile(self.cache_dir)
                if not Path(weights_path).exists() and self.cache_dir.filesystem == 's3':
                    s3_wpath = S3File(str(self.cache_dir),'checkpoints',Path(weights_path).name)
                    if s3_wpath.exists():
                        s3_wpath.download(Path(weights_path))

                if not Path(opt_weights_path).exists() and self.cache_dir.filesystem == 's3':
                    s3_opath = S3File(str(self.cache_dir),'checkpoints',Path(opt_weights_path).name)
                    if s3_opath.exists():
                        s3_opath.download(Path(opt_weights_path))
                
                if Path(weights_path).exists():
                    dienen_model.set_weights(weights_path)
                if Path(opt_weights_path).exists():
                    dienen_model.set_optimizer_weights(opt_weights_path)

        if 'extra_data' in self.parameters:
            dienen_model.set_extra_data(self.parameters['extra_data'])

        dienen_model.cache = self.cache
        if last_epoch < self.parameters['dienen_config/Model/Training/n_epochs']:
            if last_intraepoch is not None:
                train_data.set_initial_intraepoch(last_intraepoch, last_step,last_intraepoch_step, next_intra_epoch_step)
                dienen_model.core_model.model._train_counter.assign_add(last_step)

            dienen_model.fit(train_data, validation_data = validation_data, from_epoch=last_epoch, from_step=last_step, class_weights=self.parameters.get('class_weights'))

        dienen_model.load_weights(strategy='min')
        dienen_model.clear_session()

        return dienen_model

    def make_hash_dict(self):
        from paips.utils.settings import symbols
        
        self.hash_dict = copy.deepcopy(self.parameters)
        #Remove not cacheable parameters
        if not isinstance(self.hash_dict, Config):
            self.hash_dict = Config(self.hash_dict)
        if not isinstance(self.parameters, Config):
            self.parameters = Config(self.parameters)

        epochs_path = ['dienen_config/Model/Training/n_epochs', 'epochs']

        for epoch_path in epochs_path:
            if epoch_path in self.hash_dict:
                if not epoch_path.startswith('!nocache'):
                    self.hash_dict[epoch_path] = '!nocache {}'.format(self.hash_dict[epoch_path])

        _ = self.hash_dict.find_path(symbols['nocache'],mode='startswith',action='remove_value')
        _ = self.parameters.find_path(symbols['nocache'],mode='startswith',action='remove_substring')

class DienenPredict(Task):
    def process(self):
        model = self.parameters['model']

        if isinstance(model, str):
            model = joblib.load(model)

        data = self.parameters['data']
        return_targets = self.parameters.get('return_targets',True)
        deytah_process = self.parameters.get('deytah_process',None)
        deytah_keys = self.parameters.get('deytah_keys',None)
        activations = self.parameters.get('activations','output')

        group_predictions = self.parameters.get('group_predictions_by',None)
        return_as_metadata = self.parameters.get('return_as_metadata',False)
        return_as_dataframe = self.parameters.get('return_as_dataframe',False)

        batch_as_time = self.parameters.get('batch_as_axis',None)
        return_column = self.parameters.get('return_column',None)

        if return_as_metadata:
            metadata = []
            prediction_dir = Path(self.cache_dir,'predictions')
            if not prediction_dir.exists():
                prediction_dir.mkdir(parents=True,exist_ok = True)

            if group_predictions:
                groups = data[group_predictions].unique()
                for group in tqdm.tqdm(groups):
                    file_dir = Path(prediction_dir, '{}'.format(group))
                    if self.cache and file_dir.exists():
                        print('Caching {}'.format(group))
                    else:
                        data_i = data[data[group_predictions] == group]
                        start = timer()
                        if deytah_process != None:
                            data_i = deytah_process.process(data_i.to_dict('list'))
                            if deytah_keys != None:
                                data_i = [data_i[k] for k in ['audios','masks']]
                            else:
                                data_i = data_i.values()
                        end = timer()
                        print('Deytah processing: {}'.format(start-end))
                        start = timer()
                        prediction_i = model.predict(data_i,output=activations)
                        if activations != 'output':
                            prediction_i = [prediction_i[k] for k in activations]
                        else:
                            prediction_i = prediction_i.values()
                        prediction_i = np.array(prediction_i)
                        perm = list(range(prediction_i.ndim))
                        perm[0] = 1
                        perm[1] = 0
                        prediction_i = np.transpose(prediction_i,perm)
                        if batch_as_time:
                            prediction_i = np.concatenate(prediction_i,axis=batch_as_time)
                        end = timer()
                        print('Prediction: {}'.format(start-end))
                        start = timer()
                        joblib.dump(prediction_i,file_dir,compress=3)
                        end = timer()
                        print('Saving: {}'.format(start-end))
                        #np.save(file_dir,prediction_i)
                    metadata_row = data[data[group_predictions] == group].groupby('file_name').agg(np.unique)
                    metadata_row = metadata_row.reset_index()
                    metadata_row['embedding_filename'] = str(file_dir.absolute())
                    metadata.append(metadata_row)

                    return pd.concat(metadata).reset_index().drop('index',axis=1)
            else:
                metadata = copy.deepcopy(data.data)
                metadata_path = Path(self.cache_dir,'features.h5')
                with h5py.File(metadata_path, "a") as f:
                    if not isinstance(activations,list):
                        activations = [activations]
                    data.shuffle=False
                    fgen = model.predict_generator(data,output=activations)
                    existing_idxs = list(f.keys())
                    metadata_paths = {}
                    for i in tqdm.tqdm(range(len(fgen))):
                        idxs = fgen.get_indexs(i)
                        idxs_str = [str(x_i) for x_i in idxs]
                        new_idxs = ['{}/{}'.format(i,l) for i in idxs for l in activations]
                        if not all([x in f for x in new_idxs]):
                            pred = fgen[i]
                            if self.parameters.get('unpad',False):
                                original_lens = data[i][0].shape[-1] - np.argmax(data[i][0][...,::-1]!=0,axis=-1)
                                original_lens = original_lens // self.parameters['hop_size'] - (original_lens % self.parameters['hop_size'] == 0)
                            for k,v in pred.items():
                                if self.parameters.get('unpad',False):
                                    idx_pos = idxs_str.index(k.split('/')[0])
                                    v = v[:original_lens[idx_pos]]
                                if k not in f:
                                    f.create_dataset(k,data=v)
                                if k.split('/')[1] in metadata_paths:
                                    metadata_paths[k.split('/')[1]].append(k)
                                else:
                                    metadata_paths[k.split('/')[1]] = [k]
                        else:
                            for k in new_idxs:
                                if k.split('/')[1] in metadata_paths:
                                    metadata_paths[k.split('/')[1]].append(k)
                                else:
                                    metadata_paths[k.split('/')[1]] = [k]

                for k,v in metadata_paths.items():
                    metadata[k] = v
                    metadata['{}_h5file'.format(k)] =str(metadata_path)

                self.output_names = ['out','metadata_path']
                return metadata, str(metadata_path)
                    
        else:
            if isinstance(data,tuple) or isinstance(data,list):
                if not isinstance(data[0],list):
                    generator_samples = len(data[0])
                else:
                    generator_samples = len(data[0][0])
                predictions = model.predict(data[0],output=activations)
            else:           
                generator_samples = len(data._index)
                predictions = model.predict(data,output=activations)
            return_data = (predictions[:len(data.data)],)
            self.output_names = ['predictions']
            if return_targets:
                if isinstance(data,tuple) or isinstance(data,list):
                    targets = data[1]
                else:
                    check_shape = data.__getitem__(0)[1]
                    if isinstance(check_shape,list):
                        targets = np.array([data.__getitem__(i)[1][0] for i in range(len(data))])
                    else:
                        targets = np.array([data.__getitem__(i)[1] for i in range(len(data))])
                    targets = np.reshape(targets,(-1,)+targets.shape[2:])
                    targets = targets[:generator_samples]

                #assert len(targets) == len(predictions)
                self.output_names += ['targets']
                return_data += (targets,)
            
            if return_as_dataframe:
                df = data.data
                for k,v in return_data[0].items():
                    df[k] = list(v)
                self.output_names = ('out',)
                return df

            if return_column:
                return_data += (data.data[return_column],)
                self.output_names += (return_column,)

            return return_data