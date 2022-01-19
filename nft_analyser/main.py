import time
from typing import Sequence, Union, Optional, Any

import pandas as pd
import numpy as np
import keras
import tensorflow as tf
import nltk
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import reciprocal

from sklearn.pipeline import Pipeline
from nft_analyser.transformers import *
from nft_analyser import helper


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def get_vectorization_pipeline(glove_df: pd.DataFrame) -> Pipeline:
    nft_vec_pp: Pipeline = Pipeline([
        ('selectColumns', SelectColumns(columns='name')),
        ('onlyFirstCapital', CamelCaseFirstCapital()),
        ('camelToWords', CamelCaseToWords()),
        ('cleanText', CleanText(regex=r'[^a-zA-Z0-9\$]')),
        ('tokenize', Tokenize()),
        ('removeStopWords', RemoveStopWords(nltk.corpus.stopwords.words('english'))),
        ('lemmatize', Lemmatize(lemmatizer=nltk.WordNetLemmatizer())),
        ('explodeList', ExplodeList()),
        ('gloveFeatures', Vectorize(column='name', vectorization_df=glove_df, ignore_missing=True))
    ])
    return nft_vec_pp


def get_transaction_df(value_range: Optional[Sequence[float]]=None, agg: Optional[Sequence[str]]=('mean')) -> pd.DataFrame:
    trans_df = helper.get_table('transfers')
    df = trans_df[['nft_address', 'transaction_value']]
    df['transaction_value'] = df.transaction_value * 3e3 / 1e18   # To USD
    if value_range is not None:
        df = df[(df.transaction_value > value_range[0]) & (df.transaction_value < value_range[1])]
    transaction_df = df.groupby('nft_address').agg({'transaction_value': agg})
    transaction_df.columns = [c[1] for c in transaction_df.columns]
    return transaction_df


def make_analysis_df(nft_vec_df: pd.DataFrame, transaction_df: pd.DataFrame, 
                     include_nft_age: bool=False, drop_na: bool=True) -> pd.DataFrame:
    analysis_df = transaction_df.join(nft_vec_df, how='inner')
    if include_nft_age:
        time_df = helper.get_table('mints')
        time_df = time_df[['nft_address', 'timestamp']]
        time_df = time_df.groupby('nft_address').min()
        time_df = (time.time() - time_df) / (3600*24)
        analysis_df = analysis_df.join(time_df)
        if drop_na:
            analysis_df = analysis_df.dropna()
        else:
            analysis_df = analysis_df.fillna(0.0)
    return analysis_df


def train_validation_test_split(X, y):
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def hyper_neural_network(input_shape:int, output_shape:int, num_layers:int, num_neurons:int, connect_input:bool, 
                         loss_fn:str, learning_rate:float) -> keras.Model:
    input_ = keras.layers.Input(shape=(input_shape,))
    last_ = input_
    for _ in range(num_layers):
        last_ = keras.layers.Dense(num_neurons, activation='relu')(last_)
    if connect_input:
        last_ = keras.layers.Concatenate()([input_, last_])
    output_ = keras.layers.Dense(output_shape, activation='relu')(last_)
    
    model = keras.Model(inputs=[input_], outputs=[output_])
    model.compile(loss=loss_fn, optimizer=keras.optimizers.Adam(lr=learning_rate))
    return model


class CompareName:
    def __init__(self, vec_pp, pred_model, glove_df, include_age):
        self.vec_pp = vec_pp
        self.pred_model = pred_model
        self.glove_df = glove_df
        self.glove_set = set(glove_df.index)
        self.include_age = include_age

    def _get_vec_df(self, words: Sequence[str], include_age: bool=False):
        # unmappable = {w.lower() for w in words} - self.glove_set
        # if len(unmappable) != 0:
        #     raise Exception(f'The following words cannot be analyzed: {unmappable}')
        df = pd.DataFrame({'name': words}, index=words)
        df = self.vec_pp.transform(df)
        if include_age:
            df['timestamp'] = 0.0
        return df
    
    def get_value(self, words: Sequence[str]):
        df = self._get_vec_df(words, self.include_age)
        return pd.DataFrame(self.pred_model.predict(df), index=df.index, columns=['est_value'])
        
    def get_similar(self, words: Sequence[str], limit: int=10):
        df = self._get_vec_df(words)
        sim_score_df = self.glove_df @ df.T
        return sim_score_df.apply(lambda col_ss: col_ss.sort_values(ascending=False)[:limit].index.values)

    def get_similar_value(self, words: Sequence[str], limit: int=10):
        sim_df = self.get_similar(words, limit=limit)
        res_sss = {}
        for c in sim_df.columns:
            res_sss[c] = self.get_value(sim_df[c].values).sort_values(ascending=False, by='est_value').index
        res_df = pd.DataFrame(res_sss)
        res_df.index = pd.Index(data=range(1, limit+1), name='rank')
        return res_df


# all_results = {}

if __name__ == '__main__':
    default_params = {
        'glove_features': 300,
        'nft_value_range': [0.1, 1e4],      # 10c to $10k
        'value_aggregation': ['mean'],
        'include_nft_age': True,            # Takes long time (10% out of sample improvement)
        'drop_na_age': True,
        'learning_rate': 0.8,
        'epochs': 100
    }
    
    # Convert NFT Names to Vectors
    nft_df = helper.get_table("nfts").set_index('address')      # Multiple calls ok as cached at helper level
    glove_df = helper.get_glove(features=default_params['glove_features'])
    vec_pp = get_vectorization_pipeline(glove_df)
    nft_vec_df = vec_pp.fit_transform(nft_df)
    
    transaction_df = get_transaction_df(value_range=default_params['nft_value_range'],
                                        agg=default_params['value_aggregation'])
    
    # Analysis
    analysis_df = make_analysis_df(nft_vec_df=nft_vec_df, transaction_df=transaction_df,
                                   include_nft_age=default_params['include_nft_age'],
                                   drop_na=default_params['drop_na_age'])
    
    # Run Neural Network
    y_cols = default_params['value_aggregation']
    X, y = analysis_df[[c for c in analysis_df.columns if c not in y_cols]], analysis_df[y_cols]
    train_t, val_t, test_t = train_validation_test_split(X, y)
    input_shape, output_shape = train_t[0].shape[1], train_t[1].shape[1]

    model_reg = keras.wrappers.scikit_learn.KerasRegressor(hyper_neural_network,
                  input_shape=input_shape, output_shape=output_shape, 
                  num_layers=3, num_neurons=30, connect_input=True,
                  loss_fn='mae', learning_rate=default_params['learning_rate'])
    params_dist = {
        'num_layers': [1, 2, 3, 4],
        'num_neurons': np.arange(1, 100),
        'connect_input': [True, False],
        'loss_fn': ['mse'],
        'learning_rate': reciprocal(3e-4, 3e-2)
    }
    rnd_search_cv = RandomizedSearchCV(model_reg, params_dist, n_iter=10, cv=3)        
    rnd_search_cv.fit(*train_t, epochs=default_params['epochs'], 
                      validation_data=val_t,    # Used for early stoppage
                      callbacks=[keras.callbacks.EarlyStopping(patience=10)])
    print("Best Params:")
    print(rnd_search_cv.best_params_)
    print(f"Best Score: {rnd_search_cv.best_score_}")
    # Refit the best model because of bug
    best_estimator = hyper_neural_network(**rnd_search_cv.best_params_, 
                                          input_shape=input_shape, output_shape=output_shape,)
    best_estimator.fit(*train_t, epochs=default_params['epochs'],
                       validation_data=val_t,
                       callbacks=[keras.callbacks.EarlyStopping(patience=10)])
    
    tf.keras.utils.plot_model(best_estimator, show_shapes=True)
    
    model = best_estimator
    
    # Results
    # results = {}
    # results['Mean Absolute Error'] = {
    #     'Train': model.evaluate(*train_t),
    #     'Validation': model.evaluate(*val_t),
    #     'Test': model.evaluate(*test_t)
    # }
    # all_results[hyperparams['glove_features']] = results
    
    # # Quick output comparison
    # y_comp = pd.DataFrame(model.predict(X), index=pd.Index(X.index, name='index')).rename(columns={0: 'est_value'})
    # y_comp = y_comp.groupby('index').mean()
    # y_comp = y_comp.join(transaction_df.rename(columns={c: f'real_{c}' for c in transaction_df.columns}))
    # y_comp = y_comp.join(nft_df[['name']]).set_index('name')
    # y_comp = y_comp.sort_values('est_value')
    
    # Comparison of new words
    compare = CompareName(vec_pp, model, glove_df, default_params['include_nft_age'])
    words = ['Apple', 'Mango', 'Banana', 'Maid', 'Latina', 'Kittens', 'Doge', 'Shiba', 'Ape']
    compare.get_value(words).sort_values('est_value', ascending=False)
    # compare.get_similar_value(words, limit=10)

    # # Evaluate the words which have the maximum value based on input
    # tdf = vec_pp.get_subpipeline(end_node='explodeList').transform(nft_df)
    # inp_val_df = compare.get_value(tdf.name.unique()).sort_values('est_value', ascending=False)
