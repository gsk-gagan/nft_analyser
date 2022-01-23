import re 
from typing import Optional, Union, Sequence

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import nltk


class SelectColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns: Union[Sequence[str], str]=(), index: Optional[Union[Sequence[str], str]]=None):
        self.columns = [columns] if isinstance(columns, str) else columns
        self.index = index
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame]=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.index in X.columns:
            X = X.set_index(self.index)
        return X[self.columns]


class CamelCaseFirstCapital(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame]=None):
        return self
    
    @staticmethod
    def _just_first_capital(inp: str) -> str:
        if len(inp) <= 1:
            return inp
        if len(inp) == 2:
            return f'{inp[0]}{inp[1].lower()}'
        res = [inp[0]]
        last_capital = inp[0].isupper()
        next_capital = inp[2].isupper()
        for i in range(1, len(inp)-1):
            c = inp[i]
            next_capital = inp[i+1].isupper()
            res.append(c.lower() if (last_capital and next_capital) else c)
            last_capital = c.isupper()
        res.append(inp[-1].lower() if last_capital else inp[-1])
        return ''.join(res)
            
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for c in X.columns:
            X[c] = X[c].apply(lambda x: CamelCaseFirstCapital._just_first_capital(str(x)))
        return X


class CamelCaseToWords(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame]=None):
        return self
    
    @staticmethod
    def _single_case_to_word(inp: str) -> str:
        if len(inp) <= 1:
            return inp
        
        inp = inp[0].lower() + inp[1:]
        idxs = [0] + [m.start(0) for m in re.finditer(r'[A-Z]', inp)] + [len(inp)]
        return ' '.join([inp[idxs[i]: idxs[i+1]] for i in range(len(idxs)-1)])
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for c in X.columns:
            X[c] = X[c].apply(lambda x: CamelCaseToWords._single_case_to_word(str(x)))
        return X


class CleanText(BaseEstimator, TransformerMixin):
    def __init__(self, regex: str):
        self.regex = regex
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame]=None):
            return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for c in X.columns:
            X[c] = X[c].apply(lambda x: re.sub(self.regex, ' ', x.lower()))
        return X
        

class Tokenize(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame]=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for c in X.columns:
            X[c] = X[c].apply(lambda x: nltk.tokenize.word_tokenize(x))
        return X


class RemoveStopWords(BaseEstimator, TransformerMixin):
    def __init__(self, stop_words=Sequence):
        self.stop_words = stop_words
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame]=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for c in X.columns:
            X[c] = X[c].apply(lambda x: [w for w in x if w not in self.stop_words])
        return X


class Lemmatize(BaseEstimator, TransformerMixin):
    def __init__(self, lemmatizer: nltk.WordNetLemmatizer):
        self.lemmatizer = lemmatizer
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame]=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for c in X.columns:
            X[c] = X[c].apply(lambda x: [self.lemmatizer.lemmatize(w) for w in x])
        return X


class ExplodeList(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame]=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for c in X.columns:
            X = X.explode(c)
        return X


class Vectorize(BaseEstimator, TransformerMixin):
    def __init__(self, column, vectorization_df: pd.DataFrame, ignore_missing: bool=True, drop_column=True):
        self.column = column
        self.vectorization_df = vectorization_df.copy()
        self.available_values = set(vectorization_df.index)
        self.ignore_missing = ignore_missing
        self.drop_column = drop_column
        self.missing = {}
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame]=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.ignore_missing:
            df = X[X[self.column].apply(lambda x: x in self.available_values)]
            self.missing = set(X[self.column]) - set(df[self.column])
        else:   # TODO: Add options to implement similarity detection
            df = X
        df = df.join(self.vectorization_df, on=self.column)
        return df.drop(columns=self.column) if self.drop_column else df
