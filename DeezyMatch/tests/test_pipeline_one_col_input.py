#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import pytest


def test_pipeline_one_col_input():

    from DeezyMatch import train as dm_train

    # train a new model
    dm_train(
        input_file_path="./inputs/input_dfm_pytest_003.yaml",
        dataset_path="./dataset/dataset-string-matching_train.txt",
        model_name="test003",
    )

    from DeezyMatch import finetune as dm_finetune

    # fine-tune a pretrained model stored at pretrained_model_path and pretrained_vocab_path
    dm_finetune(
        input_file_path="./inputs/input_dfm_pytest_003.yaml",
        dataset_path="./dataset/dataset-string-matching_finetune.txt",
        model_name="finetuned_test003",
        pretrained_model_path="./models/test003/test003.model",
        pretrained_vocab_path="./models/test003/test003.vocab",
    )

    from DeezyMatch import inference as dm_inference

    with pytest.raises(IndexError):
        # model inference using a model stored at pretrained_model_path and pretrained_vocab_path
        # This should raise an IndexError as we need three columns during inference
        dm_inference(
            input_file_path="./inputs/input_dfm_pytest_003.yaml",
            dataset_path="./dataset/dataset-candidates.txt",
            pretrained_model_path="./models/finetuned_test003/finetuned_test003.model",
            pretrained_vocab_path="./models/finetuned_test003/finetuned_test003.vocab",
        )

    # model inference using a model stored at pretrained_model_path and pretrained_vocab_path
    dm_inference(
        input_file_path="./inputs/input_dfm_pytest_003.yaml",
        dataset_path="./dataset/dataset-string-matching_test.txt",
        pretrained_model_path="./models/finetuned_test003/finetuned_test003.model",
        pretrained_vocab_path="./models/finetuned_test003/finetuned_test003.vocab",
    )

    # Create vectors using one column input
    from DeezyMatch import inference as dm_inference

    # generate vectors for queries (specified in dataset_path)
    # using a model stored at pretrained_model_path and pretrained_vocab_path
    dm_inference(
        input_file_path="./inputs/input_dfm_pytest_003.yaml",
        dataset_path="./dataset/dataset-queries.txt",
        pretrained_model_path="./models/finetuned_test003/finetuned_test003.model",
        pretrained_vocab_path="./models/finetuned_test003/finetuned_test003.vocab",
        inference_mode="vect",
        scenario="queries_003/test",
    )

    # Create vectors using three or more columns input
    from DeezyMatch import inference as dm_inference

    # generate vectors for candidates (specified in dataset_path)
    # using a model stored at pretrained_model_path and pretrained_vocab_path
    dm_inference(
        input_file_path="./inputs/input_dfm_pytest_003.yaml",
        dataset_path="./dataset/dataset-candidates.txt",
        pretrained_model_path="./models/finetuned_test003/finetuned_test003.model",
        pretrained_vocab_path="./models/finetuned_test003/finetuned_test003.vocab",
        inference_mode="vect",
        scenario="candidates_003/test",
    )

    from DeezyMatch import combine_vecs

    # combine vectors stored in queries/test and save them in combined/queries_test
    combine_vecs(
        rnn_passes=["fwd", "bwd"],
        input_scenario="queries_003/test",
        output_scenario="combined_003/queries_test",
        print_every=10,
    )

    from DeezyMatch import combine_vecs

    # combine vectors stored in candidates/test and save them in combined/candidates_test
    combine_vecs(
        rnn_passes=["fwd", "bwd"],
        input_scenario="candidates_003/test",
        output_scenario="combined_003/candidates_test",
        print_every=10,
    )

    from DeezyMatch import candidate_ranker

    # Select candidates based on L2-norm distance (aka faiss distance):
    # find candidates from candidate_scenario
    # for queries specified in query_scenario
    candidates_pd = candidate_ranker(
        query_scenario="./combined_003/queries_test",
        candidate_scenario="./combined_003/candidates_test",
        ranking_metric="faiss",
        selection_threshold=5.0,
        num_candidates=2,
        search_size=10,
        output_path="ranker_results_003/test_candidates_deezymatch",
        pretrained_model_path="./models/finetuned_test003/finetuned_test003.model",
        pretrained_vocab_path="./models/finetuned_test003/finetuned_test003.vocab",
        number_test_rows=5,
    )

    from DeezyMatch import candidate_ranker

    # Select candidates based on conf:
    # find candidates from candidate_scenario
    # for queries specified in query_scenario
    # Verbose set as false.
    candidates_pd = candidate_ranker(
        query_scenario="./combined_003/queries_test",
        candidate_scenario="./combined_003/candidates_test",
        ranking_metric="conf",
        selection_threshold=0.5,
        num_candidates=2,
        search_size=10,
        output_path="ranker_results_003/test_candidates_deezymatch",
        pretrained_model_path="./models/finetuned_test003/finetuned_test003.model",
        pretrained_vocab_path="./models/finetuned_test003/finetuned_test003.vocab",
        number_test_rows=5,
        verbose=False,
    )

    from DeezyMatch import candidate_ranker

    # Select candidates based on L2-norm distance (aka faiss distance):
    # find candidates from candidate_scenario
    # for queries specified in query_scenario
    candidates_pd_predtrue = candidate_ranker(
        query_scenario="./combined_003/queries_test",
        candidate_scenario="./combined_003/candidates_test",
        ranking_metric="faiss",
        selection_threshold=5.0,
        num_candidates=2,
        search_size=10,
        output_path="ranker_results_003/test_candidates_deezymatch",
        pretrained_model_path="./models/finetuned_test003/finetuned_test003.model",
        pretrained_vocab_path="./models/finetuned_test003/finetuned_test003.vocab",
        number_test_rows=5,
    )

    from DeezyMatch import candidate_ranker

    # Select candidates based on L2-norm distance (aka faiss distance)
    # where ranking_metric is conf and calc_predict is false:
    candidates_pd_predfalse = candidate_ranker(
        query_scenario="./combined_003/queries_test",
        candidate_scenario="./combined_003/candidates_test",
        ranking_metric="faiss",
        selection_threshold=5.0,
        num_candidates=2,
        search_size=10,
        output_path="ranker_results_003/test_candidates_deezymatch",
        pretrained_model_path="./models/finetuned_test003/finetuned_test003.model",
        pretrained_vocab_path="./models/finetuned_test003/finetuned_test003.vocab",
        number_test_rows=5,
    )

    # Same candidates and faiss scores should be retrieved independently of calc_predict value:
    candidates_pd_predtrue.faiss_distance == candidates_pd_predfalse.faiss_distance

    from DeezyMatch import candidate_ranker
    import unicodedata

    # Select candidates based on L2-norm distance (aka faiss distance)
    # where length_diff is 2 and ranking_metric is faiss:
    candidates_pd_pred_len = candidate_ranker(
        query_scenario="./combined_003/queries_test",
        candidate_scenario="./combined_003/candidates_test",
        ranking_metric="faiss",
        selection_threshold=5.0,
        num_candidates=2,
        search_size=10,
        length_diff=2,
        calc_predict=True,
        output_path="ranker_results_003/test_candidates_deezymatch",
        pretrained_model_path="./models/finetuned_test003/finetuned_test003.model",
        pretrained_vocab_path="./models/finetuned_test003/finetuned_test003.vocab",
        number_test_rows=5,
    )

    for i, row in candidates_pd_pred_len.iterrows():
        query = row["query"]
        for candidate in row["faiss_distance"]:
            assert abs(
                len(unicodedata.normalize("NFKD", str(query)))
                - len(unicodedata.normalize("NFKD", str(candidate)))
                <= 2
            )
