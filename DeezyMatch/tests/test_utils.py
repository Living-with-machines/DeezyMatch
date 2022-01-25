#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import pytest

from DeezyMatch import utils

def test_string_split():

    # -------------------
    kwds = {"x": "py 001 $  ", 
            "tokenize": ["char"], 
            "prefix_suffix": ["|", "|"]}
    assert utils.string_split(**kwds) == ['|', 'p', 'y', ' ', '0', '0', '1', ' ', '$', ' ', ' ', '|']
    
    # -------------------
    kwds = {"x": "py 001 $  ", 
            "tokenize": ["char"],  
            "prefix_suffix": ["|", None]}
    assert utils.string_split(**kwds) == ['|', 'p', 'y', ' ', '0', '0', '1', ' ', '$', ' ', ' ']
    
    # -------------------
    kwds = {"x": "py 001 $  ", 
            "tokenize": ["char"], 
            "prefix_suffix": [None]}
    assert utils.string_split(**kwds) == ['p', 'y', ' ', '0', '0', '1', ' ', '$', ' ', ' ']
    
    # -------------------
    kwds = {"x": "py 001 $  ", 
            "tokenize": ["char"], 
            "prefix_suffix": None}
    assert utils.string_split(**kwds) == ['p', 'y', ' ', '0', '0', '1', ' ', '$', ' ', ' ']
    
    # -------------------
    # min_gram must be >= 1
    kwds = {"x": "py 001 $  ", 
            "tokenize": ["char", "ngram", "word"], 
            "min_gram": 0, 
            "max_gram": 3, 
            "prefix_suffix": ["|", "|"]}
    with pytest.raises(AssertionError):
        utils.string_split(**kwds)
    
    # -------------------
    # max_gram must be >= min_gram
    kwds = {"x": "py 001 $  ",  
            "tokenize": ["char", "ngram", "word"], 
            "min_gram": 3, 
            "max_gram": 2,  
            "prefix_suffix": ["|", "|"]}
    with pytest.raises(AssertionError):
        utils.string_split(**kwds)
    
    # -------------------
    kwds = {"x": "py 001 $  ",  
            "tokenize": ["ngram"], 
            "min_gram": 1, 
            "max_gram": 1, 
            "prefix_suffix": None}
    assert utils.string_split(**kwds) == ['p', 'y', ' ', '0', '0', '1', ' ', '$', ' ', ' ']
    
    # -------------------
    kwds = {"x": "py 001 $  ", 
            "tokenize": ["ngram", "word"], 
            "min_gram": 3, 
            "max_gram": 3, 
            "token_sep": "$", 
            "prefix_suffix": None}
    assert utils.string_split(**kwds) == ['py ', 'y 0', ' 00', '001', '01 ', '1 $', ' $ ', '$  ', 'py 001 ', '  ']
    
    # -------------------
    kwds = {"x": "py 001 $  ", 
            "tokenize": ["ngram", "word"], 
            "min_gram": 3, 
            "max_gram": 3, 
            "token_sep": "$", 
            "prefix_suffix": ["|", "|"]}
    assert utils.string_split(**kwds) == ['|py', 'py ', 'y 0', ' 00', '001', '01 ', '1 $', ' $ ', '$  ', '  |', 'py 001 ', '  ']

    # -------------------
    kwds = {"x": "py 001 $  ", 
            "tokenize": ["word"], 
            "token_sep": "default", 
            "prefix_suffix": None}
    assert utils.string_split(**kwds) == ['py', '001']

    # -------------------
    kwds = {"x": "py 001 $  ", 
            "tokenize": ["word"], 
            "token_sep": "$", 
            "prefix_suffix": None}
    assert utils.string_split(**kwds) == ['py 001 ', '  ']

def test_normalizeString():
    x = " PY _ 001 $ :)  .  .  "
    assert utils.normalizeString(x, lowercase=True, strip=True) == 'py _ 001 $ :)  .  .'
    assert utils.normalizeString(x, lowercase=False, strip=True) == 'PY _ 001 $ :)  .  .'
    assert utils.normalizeString(x, lowercase=False, strip=False) == ' PY _ 001 $ :)  .  .  '