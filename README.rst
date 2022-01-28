============
NFT Analyser
============


.. image:: https://img.shields.io/pypi/v/nft_analyser.svg
        :target: https://test.pypi.org/project/nft-analyser/

.. image:: https://img.shields.io/travis/gsk-gagan/nft_analyser.svg
        :target: https://travis-ci.com/gsk-gagan/nft_analyser

.. image:: https://readthedocs.org/projects/nft-analyser/badge/?version=latest
        :target: https://nft-analyser.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

**Package for analyzing nft database**


* Free software: MIT license
* Documentation: https://nft-analyser.readthedocs.io/.


Features
--------

* The package tries to find semantic relationship between NFT names and their current market value.
* Can be used to identify more popular NFT names from a list of specified names.
* Provides suggestions on semantically similar words which can be more popular NFT names.



Getting Started
---------------
Up and Running
^^^^^^^^^^^^^^
* Simply launch ``notebooks/main.ipynb``.
* Note that the notebook depends on heavy datasets not included with the repo. 
    * Use the links in the notebook to get them and specify the right paths.

Full Running
^^^^^^^^^^^^
* Data sources listed below are needed to run the full analysis.
    * Download NFT sqlite from: https://www.kaggle.com/simiotic/ethereum-nfts
    * Download full pre-trained GLOVE data set from: https://nlp.stanford.edu/projects/glove/
* After downloading the data sources, you need to modify config.json present in ``nft_analyser/nft_analyser/data`` sub package.
* If above modifications are not done, the tool will work with default data subset present in the package.
* You can use either ``notebooks/main.ipynb`` or ``nft_analyser/nft_analyser/main.py`` to run the full analysis. (Or ``nft_analyser/main_simple_data.ipynb`` for quick analysis)


Model Details
-------------

Introduction
^^^^^^^^^^^^
There are two key ideas of this project/package. The first one is to assess the favorablity of the NFT market for a new player from trading or creating sides. We do simple statistical analysis along with creating boxplots to get started with it.
The second key idea is to find new & popular NFT names. To do this we semantically analyze existing NFT names and transaction data. This allows us to assign a value to any word. Using this value we can compare a bunch of possible NFT names or find the one which semantically similar name will have a higher popularity in the NFT market. Because we are trying to find new names, we're interested in the out-of-sample performance of the model. Alternatively, we could easily be interested in finding the most popular NFT names/words, for which our focus will change to having the best in-sample performance. For the sake of being creative and finding new and interesting names, for this project we choose the former.

Most of the code relevant to the model lies inside ``nft_analyser/nft_analyzer/main.py``. The essential parts of the model are built as pipelines, so anyone can easily visualize the model. We've also added model pipelines for your reference here.

We can divide the project into 3 parts:
1. Vectorizing NFT Names: Converting NFT names to semantic vectors
2. Linking NFT Names to Market Value: Regressing NFT's semantic vectors against their current market value
3. Identifying New NFT Names: New NFT names comparison and identifying better names
Below we'll provide more details on the above 3 sections

Vectorizing NFT Names
^^^^^^^^^^^^^^^^^^^^^
* From initial data analysis we understood that there's a very wide range over which NFT values can range. So, we narrowed down the range of NFTs here to look only at the ones Which have a market value below $10,000. 
* Then we cleaned the NFT names to get them ready for vectorization. The cleaning process includes, splitting camelCase words to individual words and keeping alphanumeric (and $) characters. 
* Then we tokenized the names into words, removed stop words and lemmatized them before the names became ready for vectorization.
* We semantically vectorized our NFT names by using pre-trained GLOVE model as the vocabulary of names available wouldn't have been sufficient to identify any semantic relationship.
* By vectorizing we get both a numerical representation of the data, as well as a semantic relationship between different words.

.. figure:: https://github.com/gsk-gagan/nft_analyser/blob/master/docs/_static/clean.png?raw=true
    :align: center
    :height: 500px
    :alt: Vectorization Pipeline
    :figclass: align-center

    Fig 1: Cleaning and vectorization pipeline


Linking NFT Names to Market Value
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* To identify relationship between NFT names and their market values, we tried a bunch of different models like linear/polynomial regression, random forest regression, support vector machines and deep neural networks. From preliminary results, most of these models have similar results, but deep neural network were able to perform slightly better on the out of sample performance. So we defaulted to choosing that.
* We tried both GridSearchCV and RandomizedSearchCV to identify the optimal set of parameters for our deep neural network. The mean absolute error is about $700 for NFTs ranging between 10cents and $10,000. We believe having more data should help in alleviating this difference. Do note again that this is the out-of-sample error.
* From an alternative perspective, instead of focusing on the out-of-sample performance, we can reduce the limited data issue by training the model on the full data. The use case of such model will be to suggest which of the existing similar names will be best suited for creating a new NFT. 

.. figure:: https://github.com/gsk-gagan/nft_analyser/blob/master/docs/_static/model.png?raw=true
    :align: center
    :height: 500px
    :alt: Deep Neural Network
    :figclass: align-center

    Fig 2: Deep Neural Network with input layer feeding into output layer

Identifying New NFT Names
^^^^^^^^^^^^^^^^^^^^^^^^^
* There are two things which we're able to do with this package. The first one is out of a bunch of possible NFT names, find the one which has the highest possible value. The second one is, given a name find semantically similar names which can have a higher market value.
* For the first part, we start by vectorizing new NFT names and estimate their value based on our trained deep neural network.
* For the second part, we again start by vectorizing new NFT names, then compute their semantic similarity by using the GLOVE dataset. This similarity was done by calculating the Euclidean distance (or the cosine distance). Then we simply rank the top n names based on their estimated market value which was again computed using the trained deep neural network.


Findings Summary
^^^^^^^^^^^^^^^^
* The NFT market is favorable for new comers, as a vast majority of NFTs are priced reasonably and most NFT players still hold a reasonbale number of NFTs.
* The semantic model helps in identifying popular names which for the trading side can aid with making decision on which NFTs to choose and for the seller to use a semantically similar name for their works of art.
Do visit: https://medium.com/@gagandeep.singh.iitdelhi/semantic-strategy-for-entry-into-the-nft-marketplace-300b6c13fa4d for a short article on the project.


Package Details
---------------

Libraries Used
^^^^^^^^^^^^^^

Apart from the standard ones used to configure the project, the following are some noteworthy packages used:
* pandas==1.3.5 - Essential library for data analysis
* numpy==1.19.2 - Essential library for efficient numerical computation
* keras==2.3.1 - For the Neural Network Model
* tensorflow==2.0.0 - Backend of the Neural Network used by Keras
* nltk==3.6.5 - For Natuaral Language processing. Used while creating semantic features
* sklearn==1.0.2 - Essential library. Also used for creating transformers for model pipelines
* scipy==1.6.2 - Essential library. Used during RandomizedSearchCV.
* matplotlib==3.5.1 - Essential plotting library

Noteworthy Files in the Package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
These files exist in nft_analyser/*
* main.py - Contains the main code to run the entire model. It's basically similar to notebooks/main.ipynb.
* helper.py - Contains helper functions for the project. Like get the table, specify configuration etc.
* transformers.py - Contains individual transformers used within the model pipelines.


Acknowledgements
----------------
* https://www.kaggle.com/simiotic/ethereum-nfts for providing NFT dataset.
* https://nlp.stanford.edu/projects/glove/ for providing pre-trained GLOVE dataset.


Possible Issues
---------------
Package Related
^^^^^^^^^^^^^^^
* To run ``sklearn.model_selection.RandomizedSearchCV`` you need to downgrade scikit-learn to 0.21.2. There's a known bug which causes random search to fail after some runs.

Model Related
^^^^^^^^^^^^^
* It's our assumptions that NFT words should have correlation with NFTs popularity (attributed by its price).
* Because of limited data we weren't able to find a very solid model.

