Context processors directory:
- This directory contains configuration files for preprocessor of dataset text.
Dataset directory:
- This directory contains configuration files for datasets.
- Change or add dataset for document collections (dev option).
- Change or add dataset for queries collections (it must be Q&A dataset with question and answer label!).
- Train and Test configurations are by default empty but they can be configured with the datasets (just specify the datasets as in dev option) to enable fine tuning of retriever (train) and benchmarks (test).
Evaluator directory:
- This directory contains configuration files for benchmarks.
- Edit parameters to change type of evaluation and thresholds for the pipeline.
Generators directory:
- This directory contains configuration files for LLMs (generators).
- Edit the configuration parameters according to your available hardware resources.
Prompt directory:
- This directory contains configuration files for generator base prompt.
- Change or add if you want custom prompt.
Query generator directory:
 - ???
Reranker directory:
 - This directory contains configuration files for rerankers for documents retrieved in the retrieval phase.
 - Edit the configuration parameters according to your available hardware resources or create other custom files in same format.
Retriever directory:
 - This directory contains configuration files for retrieval models.
 - Edit the configuration parameters according to your available hardware resources or create other custom files in same format.
Train directory:
 - This directory contains configuration files for fine tuning.
 - Edit parameters to adjust to your dataset.
 Generated directories:
 - Indexes
 - Datasets
 - Runs
 - Experiments