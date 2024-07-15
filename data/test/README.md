Annotations used by the RAG system are in json format under `annotations`. These are exported as txt format.

In the test set, the first 60 questions are human annotated, the remaining 900 are machine generated.

Due to computation cost, the evaluation on the 900 machine generated questions is skipped every three questions (i.e. indexes 0, 3, 6, ... are used).