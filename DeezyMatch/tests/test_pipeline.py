#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import pytest

def test_train():
    from DeezyMatch import train as dm_train
    # train a new model
    dm_train(input_file_path="./inputs/input_dfm_pytest_001.yaml",
             dataset_path="./dataset/dataset-string-similarity_test.txt",
             model_name="test001")

def test_finetune():
    from DeezyMatch import finetune as dm_finetune
    # fine-tune a pretrained model stored at pretrained_model_path and pretrained_vocab_path
    dm_finetune(input_file_path="./inputs/input_dfm_pytest_001.yaml",
                dataset_path="./dataset/dataset-string-similarity_test.txt",
                model_name="finetuned_test001",
                pretrained_model_path="./models/test001/test001.model",
                pretrained_vocab_path="./models/test001/test001.vocab")

def test_inference():
    from DeezyMatch import inference as dm_inference

    # model inference using a model stored at pretrained_model_path and pretrained_vocab_path
    dm_inference(input_file_path="./inputs/input_dfm_pytest_001.yaml",
                 dataset_path="./dataset/dataset-string-similarity_test.txt",
                 pretrained_model_path="./models/finetuned_test001/finetuned_test001.model",
                 pretrained_vocab_path="./models/finetuned_test001/finetuned_test001.vocab")
    
def test_generate_query_vecs():
    from DeezyMatch import inference as dm_inference
    
    # generate vectors for queries (specified in dataset_path)
    # using a model stored at pretrained_model_path and pretrained_vocab_path
    dm_inference(input_file_path="./inputs/input_dfm_pytest_001.yaml",
                 dataset_path="./dataset/dataset-string-similarity_test.txt",
                 pretrained_model_path="./models/finetuned_test001/finetuned_test001.model",
                 pretrained_vocab_path="./models/finetuned_test001/finetuned_test001.vocab",
                 inference_mode="vect",
                 scenario="queries/test")

def test_generate_candidate_vecs():
    from DeezyMatch import inference as dm_inference
    
    # generate vectors for candidates (specified in dataset_path)
    # using a model stored at pretrained_model_path and pretrained_vocab_path
    dm_inference(input_file_path="./inputs/input_dfm_pytest_001.yaml",
                 dataset_path="./dataset/dataset-string-similarity_test.txt",
                 pretrained_model_path="./models/finetuned_test001/finetuned_test001.model",
                 pretrained_vocab_path="./models/finetuned_test001/finetuned_test001.vocab",
                 inference_mode="vect",
                 scenario="candidates/test")

def test_assemble_queries():
    from DeezyMatch import combine_vecs

    # combine vectors stored in queries/test and save them in combined/queries_test
    combine_vecs(rnn_passes=['fwd', 'bwd'], 
                 input_scenario='queries/test', 
                 output_scenario='combined/queries_test', 
                 print_every=10)

def test_assemble_candidates():
    from DeezyMatch import combine_vecs

    # combine vectors stored in candidates/test and save them in combined/candidates_test
    combine_vecs(rnn_passes=['fwd', 'bwd'],
                 input_scenario='candidates/test',
                 output_scenario='combined/candidates_test',
                 print_every=10)

def test_candidate_ranker():
    from DeezyMatch import candidate_ranker
    
    ### # Select candidates based on L2-norm distance (aka faiss distance):
    ### # find candidates from candidate_scenario
    ### # for queries specified in query_scenario
    ### candidates_pd = \
    ###     candidate_ranker(query_scenario="./combined/queries_test",
    ###                      candidate_scenario="./combined/candidates_test",
    ###                      ranking_metric="faiss",
    ###                      selection_threshold=5.,
    ###                      num_candidates=2,
    ###                      search_size=10,
    ###                      output_path="ranker_results/test_candidates_deezymatch",
    ###                      pretrained_model_path="./models/finetuned_test001/finetuned_test001.model",
    ###                      pretrained_vocab_path="./models/finetuned_test001/finetuned_test001.vocab",
    ###                      number_test_rows=20)
