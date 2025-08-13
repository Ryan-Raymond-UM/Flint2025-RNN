import json
import matplotlib.pyplot as plt
import pickle
import sklearn
import sklearn.metrics
import statistics
import torch

BATCH_SIZE=256
EMBEDDING_DIM=32
HIDDEN_SIZE=64
MAX_LEN = 16000

def collate_fn(batch):
	x, y = zip(*batch)
	y = torch.stack(y)
	x = [_x[-MAX_LEN:] for _x in x]
	x = torch.nn.utils.rnn.pad_sequence(x, True, padding_side='left')
	return x.cuda(), y.cuda()

class EmbeddedRNN(torch.nn.Module):
	def __init__(self, num_classes, num_embeddings, embedding_dim=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE):
		super().__init__()

		padding_idx = 0
		self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx)

		input_size = embedding_dim
		hidden_size = hidden_size
		self.recurrent = torch.nn.RNN(input_size, hidden_size, batch_first=True)

		in_features = hidden_size
		out_features = num_classes
		self.unembedding = torch.nn.Linear(in_features, out_features)
	def forward(self, x):
		x = self.embedding(x)
		x = self.recurrent(x)[1].squeeze()
		x = self.unembedding(x)
		return x

if __name__ == '__main__':
	with open('id2c.json', 'r') as file:
		id2c = json.load(file)
	with open('categories.json', 'r') as file:
		categories = [id2c[str(id)] for id in json.load(file)]
	with open('x.pickle', 'rb') as file:
		x = pickle.load(file)
	with open('y.pickle', 'rb') as file:
		y = pickle.load(file)

	assert(len(x) == len(y))
	num_samples = len(x)
	num_passes = 4
	split = int(num_samples / num_passes)
	train_x = x[:-split]
	test_x = x[-split:]

	train_y = y[:-split]
	test_y = y[-split:]

	print('X:', len(x), 'Train X:', len(train_x), 'Test X:', len(test_x))
	print('Y:', len(y), 'Train Y:', len(train_y), 'Test Y:', len(test_y))

	train = list(zip(train_x, train_y))
	test = list(zip(test_x, test_y))

	train_loader = torch.utils.data.DataLoader(train, BATCH_SIZE, True, collate_fn=collate_fn)
	test_loader = torch.utils.data.DataLoader(test, BATCH_SIZE, True, collate_fn=collate_fn)
	#test_loader = torch.utils.data.DataLoader(train, BATCH_SIZE, True, collate_fn=collate_fn)

	num_embeddings = max(_x.max().item() for _x in x if len(_x))+1
	num_classes = 20
	
	print('#Embeddings', num_embeddings)
	print('#Classes', num_classes)

	model = EmbeddedRNN(num_classes, num_embeddings).cuda()
	loss_fn = torch.nn.CrossEntropyLoss()
	lr = 0.01
	parameters = [{'params': model.embedding.parameters(), 'lr': lr*10},
		{'params': model.recurrent.parameters(), 'lr': lr},
		{'params': model.unembedding.parameters(), 'lr': lr}]
	optimizer = torch.optim.Adam(parameters)

	for i in range(10):
		losses = list()
		for x, y_true in train_loader:
			optimizer.zero_grad()
			y_pred = model(x)
			loss = loss_fn(y_pred, y_true)
			loss.backward()
			optimizer.step()
			losses.append(loss.item())
		print(i, 'Loss', statistics.mean(losses))
		with torch.no_grad():
			preds = list()
			truth = list()
			for x, y_true in test_loader:
				y_pred = model(x).argmax(1)
				preds.extend(y_pred.tolist())
				truth.extend(y_true.tolist())
			print(i, 'Accuracy', statistics.mean((p==t for p,t in zip(preds, truth))))
	cm = sklearn.metrics.confusion_matrix(truth, preds, normalize='true')
	disp = sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=categories)
	fig, ax = plt.subplots()
	disp.plot(ax=ax)
	plt.setp(ax.get_xticklabels(), rotation=90)
	plt.show()		
