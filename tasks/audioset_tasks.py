from paips.core import Task
import json
from pathlib import Path

class AudiosetDecodeOntology(Task):
	def process(self):
		df_data = self.parameters['in']
		if self.parameters['ontology_file'].startswith('http:/') or self.parameters['ontology_file'].startswith('https:/'):
			from urllib.request import urlopen
			with urlopen(self.parameters['ontology_file']) as url:
				ontology = json.loads(url.read().decode())
		else:
			ontology = json.load(open(str(Path(self.parameters['ontology_file']).expanduser()),'r'))
		ontology = {elem['id']: elem for elem in ontology}
		source_column = self.parameters.get('column',None)
		if source_column is None:
			raise Exception('column parameter needed. What column will be decoded?')
		destination_column = self.parameters.get('new_column',source_column)
		df_data[destination_column] = df_data[source_column].apply(lambda x: [ontology[s]['name'] for s in x.split(',')])

		return df_data