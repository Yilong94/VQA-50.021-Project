import tkinter as tk
from tkinter import *
import json
import torch
import torch.nn as nn
import h5py
import random
from PIL import Image, ImageTk
import full_model as model
import os

import data
import config
import matplotlib.pyplot as plt


def prepare_questions(questions_json):
	""" Tokenize and normalize questions from a given question json in the usual VQA format. """
	questions = [q['question'] for q in questions_json['questions']]
	for question in questions:
		question = question.lower()[:-1]
		yield question.split(' ')

def _encode_question(question, max_question_length, token_to_index):
	""" Turn a question into a vector of indices and a question length """
	vec = torch.zeros(max_question_length).long()
	for i, token in enumerate(question):
		index = token_to_index.get(token, 0)
		vec[i] = index
	return vec, len(question)

def _create_coco_id_to_index(image_features_dir):
	""" Create a mapping from a COCO image id into the corresponding index into the h5 file """
	with h5py.File(image_features_dir, 'r') as features_file:
		coco_ids = features_file['ids'][()]
	coco_id_to_index = {id: i for i, id in enumerate(coco_ids)}
	return coco_id_to_index

def _load_image(image_id, image_features_dir, coco_id_to_index):
	""" Load an image """
	features_file = h5py.File(image_features_dir, 'r')
	index = coco_id_to_index[image_id]
	dataset = features_file['features']
	img = dataset[index].astype('float32')
	return torch.from_numpy(img)

image_dir = config.val_path
image_features_dir = config.preprocessed_path
question_dir = "vqa/v2_OpenEnded_mscoco_val2014_questions.json"
# answer_dir = "vqa/v2_OpenEnded_mscoco_val2014_questions"
vocab_dir = "vocab.json"
checkpoint = "model.pth"

def main():
	with open(question_dir, 'r') as fd:
		questions_json = json.load(fd)
	with open(vocab_dir, 'r') as fd:
		vocab_json = json.load(fd)

	token_to_index = vocab_json['question']
	index_to_answer = {v:k for k,v in vocab_json['answer'].items()}

	questions =  list(prepare_questions(questions_json))
	max_question_length = max(map(len, questions))
	questions_tensor = [_encode_question(q, max_question_length, token_to_index) for q in questions]

	q_total = len(questions)
	print(questions[0])
	print(questions_tensor[0])

	loader = data.CocoImages(config.val_path).id_to_filename

	coco_id_to_index = _create_coco_id_to_index(image_features_dir)
	coco_ids = [q['image_id'] for q in questions_json['questions']]
	# Note: we can use coco id to retrieve the image and the question
	
	# load the weights into model
	log = torch.load(checkpoint)
	# get the model architecture
	vqa_model = nn.DataParallel(model.Net(len(token_to_index)+1)) # need to place model.py in the same file
	# print(log['weights'])
	vqa_model.load_state_dict(log['weights'])

	def generateSample(root, w):
		# randomly choose a question 
		rand = random.randint(0,q_total)
		sample_question = questions[rand]
		# get the corresponding coco image id after choosing question
		sample_image_id = coco_ids[rand]
		sample_image_id_pathname = loader[sample_image_id]
		#sample_image = Image.open(os.path.join(image_dir,str(sample_image_id_pathname))) # assuming that the image filenames are named based on coco image id
		#plt.imshow(sample_image)
		#plt.show()
		sample_image = ImageTk.PhotoImage(Image.open(os.path.join(image_dir,str(sample_image_id_pathname))))
		# prepare tensors for model input
		sample_question_tensor, sample_question_len = questions_tensor[rand]
		sample_image_tensor = _load_image(sample_image_id, image_features_dir, coco_id_to_index)

		# change to batch size of 1
		sample_question_tensor_batch = sample_question_tensor.unsqueeze(0)
		sample_image_tensor_batch = sample_image_tensor.unsqueeze(0)
		sample_question_len_batch = torch.LongTensor([sample_question_len]).unsqueeze(0) # not sure if this is correct
		#print(sample_question_tensor_batch)
		#print(sample_question_tensor_batch.shape)
		#print(sample_image_tensor_batch.shape)
		#print(sample_question_len_batch)
		#print(sample_question_len_batch.shape)
		out = vqa_model(sample_image_tensor_batch, sample_question_tensor_batch, sample_question_len_batch)
		_, answer = out.data.max(dim=1)
		
		# how to output the answer?hmmm....

		sample_question[0] = sample_question[0].title()
		sample_question = "Question: "+" ".join(sample_question)+"?"

		sample_answer = index_to_answer[int(answer.view(-1))]
		sample_answer = "\nPredicted Answer: "+sample_answer.title()
		caption = sample_question+sample_answer

		#w = tk.Label(root, image=sample_image, text=sample_question, compound='bottom')
		w.configure(image=sample_image, text=caption, compound='top')
		w.image = sample_image
		#b = tk.Button(root, text="Click me to sample model", command = generateSample()) # is it recursive?? 
		#b.pack()

	# initialize the tkinkter app
	root = tk.Tk()
	root.title('AI Project')
	root.geometry('800x800')
	#img = ImageTk.PhotoImage(Image.open("D:/SUTD/Term 8/Artificial intelligence/AI PROJECT/ProjectFolder/mscoco/val2014\COCO_val2014_000000040485.jpg"))
	w = tk.Label(root, compound="bottom")
	w.pack()
	b1 = tk.Button(root, text="New Picture", command = lambda:generateSample(root, w))
	b1.pack(side=BOTTOM, pady= 50)

	root.mainloop()

if __name__=='__main__':
	main()