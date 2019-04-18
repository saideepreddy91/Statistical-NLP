***[CSE 256 SP 19: Programming assignment 1: Comparing Language Models ]***

You should be able to run:
For Laplace smoothing without sampling sentence
 > python data_laplace_smoothing.py 0
With sampling a sentence 
 > python data_laplace_smoothing.py 1

For Linear interpolation without sampling sentence
 > python data_linear_interpolation.py 0
With sampling a sentence 
 > python data_linear_interpolation.py 1

You can ignore the command line argument as it does not throw any runtime error (by default no sample sentences are shown as they take longer time to run)

***[ Files ]***

There are four python files and a python notebook in this folder:

- (lm.py): This file describes the higher level interface for a language model, and contains functions to train, query, and evaluate it. An implementation of a simple back-off based unigram model is also included, that implements all of the functions
of the interface.

- (generator.py): This file contains a simple word and sentence sampler for any language model. Since it supports arbitarily complex language models, it is not very efficient. If this sampler is incredibly slow for your language model, you can consider implementing your own (by caching the conditional probability tables, for example, instead of computing it for every word).

-  (data_laplace_smoothing.py): The primary file to run the Laplace smoothing with command line argument to show the sentences sampled or not . This file contains methods to read the appropriate data files from the archive, train and evaluate all the unigram language models (by calling “lm.py”), and generate sample sentences from all the models (by calling  “generator.py”). 

-  (data_linear_interpolation.py): The primary file to run the linear interpolation with command line argument to show the sentences sampled or not . This file contains methods to read the appropriate data files from the archive, train and evaluate all the unigram language models (by calling “lm.py”), and generate sample sentences from all the models (by calling  “generator.py”). 

- (Adaptation and plots for hyperparameter tuning.ipynb): This has the adaptation code shown as a demo for different fractions of another corpus' vocabulary included into the original vocabulary. This notebook also has plots generated used in the report.

