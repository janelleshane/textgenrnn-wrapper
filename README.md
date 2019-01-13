# textgenrnn-wrapper
A python wrapper for textgenrnn to manage saved files

For more extensive documentation, see textgenrnn: https://github.com/minimaxir/textgenrnn

## Training a brand new model

To train a new model from the text file in `'data/ice-cream/IceCream_sorted.txt'` and save in weights/ice_cream:

`python textgenrnn_save_wrapper.py --model_name=ice_cream --new_model --num_epochs=1 --data_file='ice-cream/IceCream_sorted.txt'`

## Sampling from an already-trained model

To sample from the model saved in weights/ice_cream:

`python textgenrnn_save_wrapper.py --model_name=ice_cream`
`python textgenrnn_save_wrapper.py --model_name=ice_cream --n_gen=10 --temperature=0.2`
`python textgenrnn_save_wrapper.py --model_name=ice_cream --n_gen=10 --temperature=0.2 --prefix='Chocolate'`

To sample from the model saved in weights/ice_cream, but print to file instead of to the command line:

`python textgenrnn_save_wrapper.py --model_name=ice_cream --generate_to_file=ice_cream_output.txt`

## Continue training an already-trained model

Load a model from ice_cream, train on the text file in `'data/ice-cream/IceCream_sorted.txt'` and overwrite the model in ice_cream

`python textgenrnn_save_wrapper.py --model_name=ice_cream --num_epochs=1 --data_file='ice-cream/IceCream_sorted.txt'`

Load a model from ice_cream, train on the text file in 'data/ice-cream/IceCream_sorted.txt' and save in weights/ice_cream2:

`python textgenrnn_save_wrapper.py --model_name=ice_cream --num_epochs=1 --data_file='ice-cream/IceCream_sorted.txt' --save_name=ice_cream2`


## Transfer learning: 

Load a model from ice_cream, train on the text file in `'data/superheroes/superheroes.txt'` and save in weights/ice_cream_heroes:

`python textgenrnn_save_wrapper.py --model_name=ice_cream --num_epochs=1 --data_file='superheroes/superheroes.txt' --save_name=ice_cream_heroes`

## Word level:

train a word-level model and save in weights/ice_cream_word:

`python textgenrnn_save_wrapper.py --model_name=ice_cream_word --new_model --word_level --num_epochs=1 --data_file='ice-cream/IceCream_sorted.txt'`

`python textgenrnn_save_wrapper.py --model_name=ice_cream_word --new_model --word_level --max_words=10 --num_epochs=1 --data_file='ice-cream/IceCream_sorted.txt'`

Load from file:

`python textgenrnn_save_wrapper.py --model_name=ice_cream_word --temperature=1.0`

## Large text mode: 

train:

`python textgenrnn_save_wrapper.py --model_name=cocktails_largetext --large_text --new_model --num_epochs=1 --data_file='cocktails/cocktails.txt'`

Load from file:

`python textgenrnn_save_wrapper.py --model_name=cocktails_largetext --temperature=0.5 --n_gen=1 --max_gen_length=1000`
