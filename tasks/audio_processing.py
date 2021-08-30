from paips.core import Task
import tqdm
import glob
from pathlib import Path
from pyknife.audiofile import ffmpeg_to_wav
from pyknife.aws import S3File
import os
from pymediainfo import MediaInfo

class ConvertToWav(Task):
	def process(self):
		destination_dir = self.parameters.get('destination_path',None)
		if destination_dir is not None:
			destination_dir = Path(destination_dir).expanduser()
		if destination_dir is not None and not Path(destination_dir).expanduser().exists():
			destination_dir.mkdir(parents=True)
		
		sr = self.parameters.get('sr',None)

		if not isinstance(self.parameters['filenames'],list):
			filenames = [self.parameters['filenames']]
		else:
			filenames = self.parameters['filenames']
		converted = []
		for file in tqdm.tqdm(filenames):
			file = Path(file)
			if destination_dir:
				destination_path = Path(destination_dir,'{}.wav'.format(file.stem))
			else:
				destination_path = Path(file.parent,'{}.wav'.format(file.stem))
			if destination_path.exists():
				info = MediaInfo.parse(str(destination_path.absolute()))
				audiotrack = None
				for track in info.tracks:
					if track.track_type == 'Audio':
						audiotrack = track
						break
				if getattr(audiotrack,'sampling_rate') == sr:
					converted.append(destination_path)
				else:
					try:
						ffmpeg_to_wav(file,destination_path,sr=sr)
						converted.append(destination_path)
					except Exception as e:
						print(e)
			else:
				try:
					ffmpeg_to_wav(file,destination_path,sr=sr)
					converted.append(destination_path)
				except Exception as e:
					print(e)

				if self.parameters.get('delete_original',False) and Path(file).exists():
					os.remove(str(file))
			
		return converted