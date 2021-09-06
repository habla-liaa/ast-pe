from paips.core import Task
import tqdm
import glob
import os
import shutil
from pathlib import Path
import pandas as pd
import zipfile
import numpy as np

import itertools
import random

from pyknife.aws import download_s3, S3File
import requests

class CalculateClassWeights(Task):
    def process(self):
        class_weights = self.parameters.get('class_weights',None)
        column = self.parameters.get('column',None)
        data = self.parameters.get('in',None)

        if class_weights == 'balanced':
            props = data[column].value_counts().sort_index()
            class_weights = len(data)/(props*len(props))
            class_weights = class_weights.values
            class_weights = {k:v for k,v in enumerate(class_weights)}
            
        return class_weights

class Concatenate(Task):
    def process(self):
        return np.concatenate(self.parameters['in'],axis=self.parameters['axis'])

class CopyFiles(Task):
    #Can copy from S3 but not viceversa
    def process(self):
        if not Path(self.parameters['destination_folder']).expanduser().exists():
            Path(self.parameters['destination_folder']).expanduser().mkdir(parents=True)
        copied = []
        if not isinstance(self.parameters['source_files'],list):
            self.parameters['source_files'] = [self.parameters['source_files']]
        for fname in tqdm.tqdm(self.parameters['source_files']):
            if fname.startswith('s3://'):
                fdir = fname.split('//')[1]
                file_parts = fdir.split('/')
                file_path = '/'.join(file_parts[1:])
                if self.parameters['destination_folder'].startswith('s3://'):
                    download_s3(file_parts[0],file_path,'s3temp')
                    destination_file = S3File(self.parameters['destination_folder'],file_path)
                    destination_file.upload('s3temp')
                    copied.append(str(destination_file))
                else:
                    destination_file = Path(self.parameters['destination_folder'],file_path).expanduser()
                    download_s3(file_parts[0],file_path,str(destination_file))
                    copied.append(str(destination_file))
            else:
                fname = Path(fname).expanduser()
                if self.parameters['destination_folder'].startswith('s3://'):
                    destination_file = S3File(self.parameters['destination_folder'],fname.name)
                    destination_file.upload(fname)
                    copied.append(str(destination_file))
                else:
                    destination_file = Path(self.parameters['destination_folder'],fname.name).expanduser()
                    shutil.copyfile(fname,str(destination_file))
                    copied.append(str(destination_file))
        self.output_names = ['parent_folder', 'copied_files']

        return Path(self.parameters['destination_folder']).parent, copied

class CycleList(Task):
    def process(self):
        cycle = self.parameters['cycle']
        out_len = len(self.parameters['out_shape'])
        out = cycle*(out_len//len(cycle)) + cycle[:(out_len%len(cycle))]

        return out

class Downloader(Task):
    def process(self):
        to_download = self.parameters['downloads']
        downloaded_files = []
        for download_metadata in to_download:
            destination_file = Path(download_metadata['DestinationPath']).expanduser()
            if not destination_file.parent.exists():
                destination_file.parent.mkdir(parents=True)
            if not destination_file.exists():
                if download_metadata['SourcePath'].startswith('http:/') or download_metadata['SourcePath'].startswith('https:/'):
                    response = requests.get(download_metadata['SourcePath'], stream = True)
                    download_file = open(str(destination_file),'wb')
                    if 'Content-Length' in response.headers:
                        pbar = tqdm.tqdm(unit="MB", total=int(response.headers['Content-Length'])/(1024*1024))
                    else:
                        pbar = None
                    for chunk in tqdm.tqdm(response.iter_content(chunk_size=1024*1024)):
                        if chunk:
                            if pbar:
                                pbar.update(len(chunk)/(1024*1024))
                            download_file.write(chunk)
                    download_file.close()
                elif download_metadata['SourcePath'].startswith('s3:/'):
                    from pyknife.aws import download_s3
                    bucket_name = download_metadata['SourcePath'].split('s3://')[-1].split('/')[0]
                    bucket_path = '/'.join(download_metadata['SourcePath'].split('s3://')[-1].split('/')[1:])
                    download_s3(bucket_name,bucket_path,str(destination_file))
            else:
                #if self.logger:
                #    self.logger.info('Skipping download as {} already exists'.format(str(destination_file)))
                #else:
                #    print('Skipping download as {} already exists'.format(str(destination_file)))
                pass
            if download_metadata.get('Extract',False) == True:
                if destination_file.suffix == '.gz':
                    import tarfile
                    if self.logger:
                        self.logger.info('Extracting downloaded files')
                    else:
                        print('Extracting downloaded files')
                    tar = tarfile.open(destination_file)
                    tar_members = tar.getnames()
                    for member in tqdm.tqdm(tar_members):
                        if not Path(destination_file.parent,member).expanduser().exists():
                            tar.extract(member,destination_file.parent)
                    tar.close()
                elif destination_file.suffix == '.zip':
                    import zipfile
                    zip_file = zipfile.ZipFile(destination_file)
                    zip_members = zip_file.namelist()
                    for member in tqdm.tqdm(zip_members):
                        if not Path(destination_file.parent,member).expanduser().exists():
                            zip_file.extract(member,destination_file.parent)
                    zip_file.close()
                downloaded_files.append(str(destination_file.parent))
            else:
                downloaded_files.append(str(destination_file))

        return downloaded_files

class GlobWrapper(Task):
    def list_files_in_folder(self, folder, expression='*'):
        if folder.startswith('s3://'):
            import boto3
            from pyknife.aws import get_all_s3_objects
            from pathlib import PurePath

            folder = folder.split('//')[1]
            parent_parts = folder.split('/')

            s3_client = boto3.client('s3')
            all_files = [k['Key'] for k in get_all_s3_objects(s3_client, Bucket=parent_parts[0], Prefix='/'.join(parent_parts[1:]))]
            expression = '/'.join(parent_parts[1:]) + expression
            dirs = [k for k in all_files if PurePath(k).match(expression)]
            dirs = ['s3://' + parent_parts[0] + '/' + d for d in dirs]
        else:
            expression = folder + expression
            dirs = glob.glob(expression)

        return dirs


    def process(self):
        parent_dir = self.parameters.get('parent_dir',None)
        max_quantity = self.parameters.get('max',None)
        if parent_dir and not parent_dir.startswith('s3://'):
            parent_dir = str(Path(parent_dir).expanduser().absolute()) + '/'
        elif parent_dir is None:
            parent_dir = ''
        print(parent_dir)

        dirs = self.list_files_in_folder(parent_dir, expression = self.parameters.get('expression','*'))
        
        filename_not_in = self.parameters.get('filename_not_in',False)
        if filename_not_in:
            not_in_paths = self.list_files_in_folder(filename_not_in, expression = self.parameters.get('expression','*'))
            filenames_not_in = [n.split('/')[-1] for n in not_in_paths]
            remaining_dirs = [d for d in dirs if d.split('/')[-1] not in filenames_not_in]
            dirs = remaining_dirs

        if max_quantity:
            dirs = dirs[:max_quantity]
        return dirs

class Group(Task):
    def process(self):
        n = self.parameters['n']
        x = self.parameters['in']
        self.output_names = ['out','len']
        reps = self.parameters.get('repetitions',1)

        structure = [n]*(len(x)//n)
        if len(x)%n > 0:
            structure = structure + [len(x)%n]
        
        offsets = list(itertools.accumulate([0] + structure[:-1]))
        groups = [x[i:i+s] for i,s in zip(offsets,structure)]
        if self.parameters.get('shuffle_repetitions',False):
            random.seed(self.parameters.get('seed',1234))
            groups_ = []
            for i in range(reps):
                groups_ = groups_ + random.sample(groups,len(groups))
            groups = groups_
        else:
            groups = groups*reps

        return groups, len(groups)

class Identity(Task):
    def process(self):
        return self.parameters['in']

class LabelEncoder(Task):
    def process(self):
        df_data = self.parameters['in']
        column = self.parameters['column']
        new_column = self.parameters.get('new_column','target')
        possible_labels = list(sorted(df_data[column].unique()))
        mapping = {j:i for i,j in enumerate(possible_labels)}
        df_data[new_column] = df_data[column].apply(lambda x: mapping[x])

        self.output_names = ['out','labels']

        return df_data, possible_labels

class Merge(Task):
    def process(self):
        in_tasks = self.parameters['in']
        if len(in_tasks)>1:
            df_merged = in_tasks[0]
            for df_i in in_tasks[1:]:
                cols = df_i.columns.difference(df_merged.columns)
                df_merged = pd.concat([df_merged,df_i[cols]],axis=1)
            return df_merged
        elif len(in_tasks)==1:
            return in_tasks[0]
        else:
            return None

class MultiHotVector(Task):
    def process(self):
        if self.parameters.get('class_map') is not None:
            class_map = pd.read_csv(self.parameters['class_map'])
            class_map = class_map.set_index(self.parameters['class_map_key_column'])
        def label_to_multi(x):
            hv = np.zeros((len(class_map),))
            for x_i in x.split(','):
                if x_i in class_map.index:
                    hv[int(class_map.loc[x_i][self.parameters['class_map_index_column']])]=1
            return hv

        self.parameters['in'][self.parameters['column_out']] = self.parameters['in'][self.parameters['column_in']].apply(label_to_multi)

        return self.parameters['in']

class Pool(Task):
    def process(self):
        pool_type = self.parameters.get('type','mean')
        axis = self.parameters.get('axis',-1)
        data = self.parameters.get('in',None)

        if self.parameters.get('enable',True):
            if pool_type == 'mean':
                return np.mean(data,axis=axis)
            elif pool_type == 'sum':
                return np.sum(data,axis=axis)
            elif pool_type == 'argmax':
                return np.argmax(data,axis=axis)
            elif pool_type == 'max':
                return np.max(data,axis=axis)
        else:
            return data

class Relabel(Task):
    def process(self):
        enable = self.parameters.get('enable',True)
        data = self.parameters['in']
        relabels = self.parameters['relabels']

        if enable:
            for relabel in relabels:
                if 'column' in relabel:
                    if 'old_name' in relabel:
                        if not isinstance(relabel['old_name'],list):
                            relabel['old_name'] = [relabel['old_name']]
                        data.loc[data[relabel['column']].isin(relabel['old_name']),relabel['column']] = relabel['new_name']
                    elif 'mapping' in relabel:
                        data[relabel['column']] = data[relabel['column']].apply(lambda x: relabel['mapping'][x])
        return data

class RemoveFiles(Task):
    def process(self):
        folder_to_remove = Path(self.parameters['folder']).expanduser()
        if folder_to_remove.exists():
            shutil.rmtree(str(folder_to_remove))

        self.output_names = ['parent_folder']
        return Path(self.parameters['folder']).parent

class ZipExtractor(Task):
    def process(self):
        extracted_files = []
        destination_path = str(Path(self.parameters.get('destination_path',None)).expanduser().absolute())
        keep_zip = self.parameters.get('keep_zip',True)
        for zip_filename in tqdm.tqdm(self.parameters['in']):
            zip_file = zipfile.ZipFile(str(Path(zip_filename).expanduser().absolute()))
            if destination_path is None:
                destination_path_ = str(Path(Path(zip_filename).parent),Path(zip_filename).stem)
            else:
                if not Path(destination_path).expanduser().exists():
                    Path(destination_path).expanduser().mkdir(parents=True)
                destination_path_ = str(Path(destination_path).expanduser())

            for member in zip_file.namelist():
                if Path(destination_path_,member).exists():
                    #print('{} already exists'.format(member))
                    pass
                else:
                    zip_file.extract(member,destination_path_)
                extracted_files.append(str(Path(destination_path_,member)))

            if not keep_zip:
                os.remove(zip_filename)

            zip_file.close()

        return extracted_files

class ZipFiles(Task):
    def get_arcname(self,f):
        if self.discard_folder_structure:
            arcname = Path(f).name
        elif self.relative_to:
            arcname = str(Path(f).relative_to(self.relative_to))
        return arcname

    def process(self):
        zip_fname = str(Path(self.parameters['zipfile']).expanduser())
        zipObj = zipfile.ZipFile(zip_fname, 'w')
        self.discard_folder_structure = self.parameters.get('discard_folder_structure',True)
        self.relative_to = self.parameters.get('relative_to',None)
        if self.relative_to:
            self.relative_to = Path(self.relative_to).expanduser()
            self.discard_folder_structure=False
        
        # Add multiple files to the zip
        if not isinstance(self.parameters['in'],list):
            self.parameters['in'] = [self.parameters['in']]
        for f in tqdm.tqdm(self.parameters['in']):

            if isinstance(f,list): #(Parche)
                for f_i in f:
                    zipObj.write(f_i,arcname=self.get_arcname(f_i))
            elif f is not None:
                zipObj.write(f,arcname=self.get_arcname(f))
            
        zipObj.close()

        return zip_fname
