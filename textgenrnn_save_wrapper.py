import shutil, os
import argparse 
from textgenrnn import textgenrnn

# --------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default='test',
					help="name of model & folder where it will be saved")
parser.add_argument("--new_model", default=False,
					help="Include to start training from scratch. Otherwise, will try to load a model from weights/model_name", action="store_true")
parser.add_argument("--config_path", default='',
					help="Location of configuration file to load for new models. Otherwise, will use default.")
parser.add_argument("--num_epochs", type=int, default=0,
					help="number of times training process will go through the entire training dataset. if 0, just generates samples.")
parser.add_argument("--data_file", help="Location of training data file. Assumed to be in data folder")
parser.add_argument("--large_text", default=False,
					help="Include to train in longtext mode instead of treating each line as a new item", action="store_true")
parser.add_argument("--word_level", default=False,
					help="Include to train in word mode rather than character mode", action="store_true")
parser.add_argument("--max_words", type=int, default=20000,
					help="maximum number of words to include in vocab for word mode")
parser.add_argument("--save_name", default = '0',
					help="Include to save the model in a new location when done. Otherwise, will save in weights/model_name")
parser.add_argument("--n_gen", type=int, default=10,
					help="number of output samples to generate")
parser.add_argument("--temperature", type=float, default=0.5,
					help="temperature to use when generating samples")
parser.add_argument("--prefix", default='',
					help="Starting text to use for sample generation")
parser.add_argument("--max_gen_length", type=int, default=1000,
					help="For sampling large_text models, the max length of a single sample")
parser.add_argument("--generate_to_file", default='',
					help="Specify a filename to generate to file instead of command line")

# ---------------------------------------------------------------------------------------

# -----  Parse input data
args = parser.parse_args()

new_model = args.new_model
if not args.new_model:
	load_loc = os.path.join('weights', args.model_name)
	
num_epochs = args.num_epochs
if num_epochs !=0:
	data_loc = os.path.join('data', args.data_file)
	if args.save_name == '0':
		save_name = args.model_name
	else:
		save_name = args.save_name

if args.word_level:
	word_level=True
	max_words = args.max_words
else:
	word_level=False

large_text = args.large_text

n_gen = args.n_gen
temperature = args.temperature
prefix = args.prefix
max_gen_length = args.max_gen_length
config_path = args.config_path
gen_path = args.generate_to_file

# new_model = True
# load_loc = 'weights/ice_cream'	#use if new_model=False
# data_loc = 'Combined_DnD_spells.txt'
# num_epochs = 1 #set to 0 to load & sample a saved model w/o training
# save_name = 'dnd' #results will be saved in textgenrnn/weights/saveloc

# ---------------------------------------------------------------------------------------

if not num_epochs == 0:

	if new_model:
		# -- start a new model (required if you want to save results) --

		# start with default weights (these will be overwritten by the new model)-
		my_model = textgenrnn(config_path=config_path)

		# train from file (will output training progress to the command line)
		if word_level:
			print('Using word-level mode.')
			my_model.train_from_file(data_loc, num_epochs=num_epochs, new_model=True, word_level=True, max_words=max_words)
		elif large_text:
			my_model.train_from_largetext_file(data_loc, num_epochs=num_epochs, new_model=True)
		else:
			my_model.train_from_file(data_loc, num_epochs=num_epochs, new_model=True)


	# -- resume a saved model --
	# note that this can a model trained on a different dataset
	if not new_model:
		# 	start with saved weights
		weights_loc = os.path.join(load_loc,'textgenrnn_weights.hdf5')
		vocab_loc = os.path.join(load_loc,'textgenrnn_vocab.json')
		config_loc = os.path.join(load_loc,'textgenrnn_config.json')
		my_model = textgenrnn(weights_path=weights_loc, vocab_path=vocab_loc, config_path=config_loc)

		# save results before training in case user aborts
		save_dir = os.path.join('weights', save_name)
		if not os.path.exists(save_dir):
			os.mkdir(save_dir)
		files = ['textgenrnn_weights.hdf5', 'textgenrnn_vocab.json', 'textgenrnn_config.json']
		for f in files:
			shutil.copy(f, save_dir)

		# train from file (will output training progress to the command line)
		if word_level:
			print('Using word-level mode.')
			my_model.train_from_file(data_loc, num_epochs=num_epochs, new_model=False, word_level=True, max_words=max_words)
		elif large_text:
			my_model.train_from_largetext_file(data_loc, num_epochs=num_epochs, new_model=False)
		else:
			my_model.train_from_file(data_loc, num_epochs=num_epochs, new_model=False)

	# save results again after training
	save_dir = os.path.join('weights', save_name)
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	files = ['textgenrnn_weights.hdf5', 'textgenrnn_vocab.json', 'textgenrnn_config.json']
	for f in files:
		shutil.copy(f, save_dir)

else:
	# num_epochs=0 so all we do is load an existing model and sample.
	weights_loc = os.path.join(load_loc,'textgenrnn_weights.hdf5')
	vocab_loc = os.path.join(load_loc,'textgenrnn_vocab.json')
	config_loc = os.path.join(load_loc,'textgenrnn_config.json')
	my_model = textgenrnn(weights_path=weights_loc, vocab_path=vocab_loc, config_path=config_loc)

	if gen_path=='':
		my_model.generate(n=n_gen, temperature=temperature, prefix=prefix, max_gen_length=max_gen_length)
	else:
		my_model.generate_to_file(gen_path, n=n_gen, temperature=temperature, prefix=prefix, max_gen_length=max_gen_length)