import concurrent.futures
import ipaddress
import json
import pandas
import pathlib
import pickle
import torch

ROOT = pathlib.Path('Flint2025-IP')

def get_remote(packet):
	"""
	If a packet is incoming or outgoing, returns the remote IP address.
	If the packet is internal, returns None.
	"""
	source = ipaddress.ip_address(packet['Source'])
	destination = ipaddress.ip_address(packet['Destination'])

	if source.is_private and destination.is_global:
		return str(destination)
	if source.is_global and destination.is_private:
		return str(source)

def df_to_list(df):
	"""
	Convert packet list df to remote list
	"""
	l = list()
	for i, packet in df.iterrows():
		remote = get_remote(packet)
		if remote and ((not l) or (l[-1] != remote)):
			l.append(remote)
	return l

def record_to_list(record):
	"""
	For use with ProcessPoolExecutor.
	"""

	csv_path = ROOT/record['packets']
	df = pandas.read_csv(csv_path)
	return df_to_list(df)

def record_to_vocab(record):
	return set(record_to_list(record))

if __name__ == '__main__':
	"""domains = pandas.read_csv('/data/ryanray/paper/domain-lists/900,000-categorized.csv')
	domain_to_category = {r['domain']:r['category'] for _,r in domains.iterrows()}
	with open('domain_to_category.json', 'w') as file:
		json.dump(domain_to_category, file)
	"""
	metadata = list()
	for i in range(1, 6):
		with open(ROOT/f'{i}.jsonl', 'r') as file:
			metadata.extend(map(json.loads, file))
	print('Done', len(metadata))
	metadata = metadata[:]
	with open('domain_to_category.json', 'r') as file:
		domain_to_category = json.load(file)
	categories = sorted(list(set(domain_to_category[r['domain']] for r in metadata)))
	with open('categories.json', 'w') as file:
		json.dump(categories, file)
	#with open('categories.json', 'r') as file:
	#	categories = json.load(file)

	y = [categories.index(domain_to_category[record['domain']]) for record in metadata]
	y = torch.LongTensor(y)
	print(y)
	#categories = sorted(list(set(domain_to_category[r['domain']] for r in metadata)))
	#with open('categories.json', 'w') as file:
	#	json.dump(categories, file)

	with concurrent.futures.ProcessPoolExecutor(max_workers=64) as e:
		vocab = set(('[PAD]')).union(*e.map(record_to_vocab, metadata))
		addresses = sorted(list(vocab), reverse=True)
		address2id = {a:i for i, a in enumerate(addresses)}
	
		lists = e.map(record_to_list, metadata)
		embedded = map(lambda l: [address2id.get(a, 0) for a in l], lists)
		x = [torch.LongTensor(e) for e in embedded]
	with open('x.pickle', 'wb') as file:
		pickle.dump(x, file)
	with open('y.pickle', 'wb') as file:
		pickle.dump(y, file)
	
