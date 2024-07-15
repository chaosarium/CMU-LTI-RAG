# ANLP HW2: A Retrieval Augmented Generation (RAG) System from Scratch

A RAG system built from scratch to answer questions about Carnegie Mellon University (CMU) and the Language Technologies Institute (LTI). Our system performed the best among all submissions in the Spring 2024 iteration of CMU's 11-711 Advanced NLP course.

What the system does by examples:

> **Q:** Who is sponsoring the event Carnival Activities Tent at CMU's Spring Carnival 2024?  
> **A:** The Spring Carnival Committee and the CMU Alumni Association.

> **Q:** Who is the first author of the paper called "Unlimiformer: Long-Range Transformers with Unlimited Length Input"?  
> **A:** Amanda Bertsch is the first author of the paper "Unlimiformer: Long-Range Transformers with Unlimited Length Input".

# Set up

To set up the repository and dependencies. Clone and `cd` into this repo, then:

```sh
git submodule update --init
pip install ragatouille rank_bm25 langchain sqlalchemy pypdf langchain_together rouge matplotlib chromadb stanza
pip uninstall --y faiss-cpu & pip install faiss-gpu
```

You'll need TogetherAI tokens in `.env`

# Pipeline Configurations

- Retrievers can be:
    - `bm25`
    - `ragatouille` (wrapper around ColBERT)
    - `vec` (uses e5-large-v2 as embedding model)
- Generators can be:
    - `llama13b`
    - `llama70b`
    - `mixtral8x7b`
    - `gemma7b`
- Additional things to toggle
    - `--hypothetical` use includes hypothetical documents in retrieval queries
    - `--chunkaugmentation` adds metadata like file names to text chunks to provide additional context [^chunkaugmentationfoot]
- Additional steps in the pipeline
    - `src/postprocess.ipynb` will pass the generated answers back to an LLM to make it more concise ("answer extraction"). This is not intended to change the content of the answer, but to fix the format of the answer if concision is needed.

[^chunkaugmentationfoot]: This was supposed to be always enabled, but a bug made it not the case. We did not test what happens when this is enabled, In theory, it should yield better performance.

## Selected configurations

Below are 3 selected systems that tend to perform well:

### System 1: k=10, ColBERT, Llama70B-chat generator, no hypothetical document

Run:

```sh
python main.py --mode evaluate --dataset txt --k 10 --testset human --generator llama70b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 10 --testset template --generator llama70b --retriever ragatouille
```

Should get:

| Testset | Mean Exact Match | Mean F1 Score | Mean Recall | Mean Precision |
| --- | --- | --- | --- | --- |
| Human Annotated | 0.08333333333333333 | 0.3324801411918672 | 0.8763468013468014 | 0.24146019628325108 |
| Machine Annotated  | 0.056666666666666664 | 0.3552887243053707 | 0.9340235976615825 | 0.25724248883603323 |

### System 2: k=10, ColBERT, Llama70B-chat generator, no hypothetical document, plus Llama70B (non-chat) answer extractor

Run answer extraction with `src/postprocess.ipynb` using Llama70B (non-chat) extractor based on answers by:

- k=10, ColBERT, Llama70B-chat generator, no hypothetical document for the Human Annotated test set.
- k=10, ColBERT, Mixtral8x7B-chat generator, no hypothetical document for the Human Annotated test set.

Note we 

Should get:

| Testset | Mean Exact Match | Mean F1 Score | Mean Recall | Mean Precision |
| --- | --- | --- | --- | --- |
| Human Annotated | 0.6833333333333333 | 0.8093650793650794 | 0.8309764309764309 | 0.8105054302422724 |
| Machine Annotated  | 0.6166666666666667 | 0.8075791959199469 | 0.8338876394492134 | 0.8119030599080405 |

The experiment outputs will be written to `experiment-alt`. We also print out before vs after extraction differences like this:

```
Before:
   Mean Exact Match  Mean F1 Score  Mean Recall  Mean Precision
0          0.083333        0.33248     0.876347         0.24146
After:
   Mean Exact Match  Mean F1 Score  Mean Recall  Mean Precision
0          0.683333       0.809365     0.830976        0.810505
```

# Data

## Raw data for retrieval

Data that could be retrieved are in the `collection` directory.

- Courses @ CMU
    - **Course data** is organized as one document per course per semester, located in `collection/courses`. Courses are covered F23 - M24. This is obtained using the `course-api` library
    - Each **academic calendar** (2324, 2425) is one document, located in `collection/academic_calendar`. This is converted and cleaned based on the excel course calendars.
- CMU events
    - Carnival located in `collection/events/carnival`
        - Schedules are scrapped using beautiful soup and are under `collection/events/carnival/events`
        - Other ones are scrapped manually
    - Commencement located in `collection/events/commencement`
        - All scrapped manually
- History
    - Buggy scrapped manually in `collection/history/buggy`
    - Athletics scrapped manually in `collection/history/athletics`
    - CMU information scrapped with beautifulsoup4 in `collection/history/cmu`
- Academics @ LTI
    - `collection/programs/handbooks` contains one `.txt` file per section per program handbook, for all program handbooks linked in the initial version of this assignment. The first line specifies the name of the handbook which includes the name of the degree program. Text files are obtained by reading handbook PDFs through `pypdf`, and egregious spacing or line break errors are fixed manually.
    - `collection/programs/webpages` contains one `.txt` file for each LTI degree program. Each file contains the information found on that degree program's section of the LTI Academics webpage. We obtained this by using `requests` to scrape the old LTI website.
- Faculty @ LTI
   - `collection/directory` contains one `.txt` file per faculty member, obtained using the `requests` library on the old LTI website.
   - `collection/papers` contains one `.txt` file per open-access paper, for all open-access papers published in 2023 on which one of the faculty members in `collection/directory` was an author. The metadata for each paper is obtained through the Semantic Scholar API.

## Question-and-Answer Annotations

### Human annotation

We manually annotate 60 questions with the goal of covering as many of the above areas as possible. The breakdown of human-annotated question categories is as follows:

- `annotations/leon-annotated.json`
    - 10 on CMU history
    - 10 on LTI programs
    - 10 on CMU events
- `annotations/karina-annotated.json`
    - 15 on CMU faculty
    - 15 on papers

### Machine annotation

We use human-crafted templates to generate questions for structured data, such as courses, papers, and faculty. The machine-generated datasets are `papersQAs`, `coursesQAs`, and `directoryQAs`, each containing 300 question-answer pairs.

### Annotation split

All of the above annotations are used for testing and are exported to `data/test`. As a small set of questions for in-context learning, we manually crafted 4 more question-answer pairs. These are reproduced in `data/train`[^1].

[^1]: We noticed later that the exact same question "What are Graham Neubig's research interests?" was created twice by both machine and human annotators. This question is thus unfortunately used for both in-context learning and evaluation, so our result may be off by 0.33%.

## Question augmentation

We augment questions (aka queries) by asking an LLM to generate a hypothetical document that would help answer the question. Query augmentation is done in `query_augment.ipynb`. 

The expected count of augmented entries for each test set (note toyQAs-on-papers.json is just for sanity checking whether code runs):

```
augmented 7 for leon-annotated
augmented 0 for karina-annotated
augmented 74 for papersQAs
augmented 22 for coursesQAs
augmented 56 for directoryQAs
```

# Running the Pipeline

## QA REPL

Run an interactive question-answering REPL. 

Here's an example of how to run the REPL to answer questions about papers by CMU faculties:

```sh
python main.py --mode repl --dataset papers --retriever bm25 --k 10
```

What to expect:

```txt
...
100%|██████████████████████████████████████████████████████| 183/183 [00:00<00:00, 5394.55it/s]
this set contains 183 documents with max len 3650
split into 715 documents with max len 649
...
Enter question:
```

And if you send questions:

```txt
Enter question: What is the publication venue of the paper titled "A Vector Quantized Approach for Text to Speech Synthesis on Real-World Spontaneous Speech"?
Answer: 
The publication venue of the paper titled "A Vector Quantized Approach for Text to Speech Synthesis on Real-World Spontaneous Speech" is AAAI Conference on Artificial Intelligence.
Enter question: Who is the first author of the paper called "Unlimiformer: Long-Range Transformers with Unlimited Length Input"?
Answer: 
Amanda Bertsch is the first author of the paper "Unlimiformer: Long-Range Transformers with Unlimited Length Input".
```

Note this is slow if retrieving from large corpus without an indexing stage. It is not recommended to use the REPL for large experiments.

## Staged Workflow

This will separate the retrieval and generation steps and is the recommended way to experiment with the system while saving time rerunning stuff.

### 1. Retrieve and save retrieval results

Note any `--generator` argument will be ignored since retrieve mode skips generation.

Staging the retrieval results with k = 10 will allow for generation experiments for any k <= 10, since we can just take the top-k predictions as long as they are available in the cached retrieval results.

`cd` into `src` and run:

```sh
python main.py --mode retrieve --dataset txt --k 10 --testset human --retriever bm25
python main.py --mode retrieve --dataset txt --k 10 --testset human --retriever bm25 --hypothetical
python main.py --mode retrieve --dataset txt --k 10 --testset human --retriever vec
python main.py --mode retrieve --dataset txt --k 10 --testset human --retriever vec --hypothetical
python main.py --mode retrieve --dataset txt --k 10 --testset human --retriever ragatouille
python main.py --mode retrieve --dataset txt --k 10 --testset human --retriever ragatouille --hypothetical
python main.py --mode retrieve --dataset txt --k 10 --testset template --retriever bm25
python main.py --mode retrieve --dataset txt --k 10 --testset template --retriever bm25 --hypothetical
python main.py --mode retrieve --dataset txt --k 10 --testset template --retriever vec
python main.py --mode retrieve --dataset txt --k 10 --testset template --retriever vec --hypothetical
python main.py --mode retrieve --dataset txt --k 10 --testset template --retriever ragatouille
python main.py --mode retrieve --dataset txt --k 10 --testset template --retriever ragatouille --hypothetical

python main.py --mode retrieve --dataset papers --k 10 --testset human --retriever bm25
```

Running the above will create cached retrieval results in the `experiment` directory. The cached pickle files are already included in this repo for our annotations.

## 2. Once retrieval results are in the experiment directory, run and evaluate generation


Since retrieval results are cached, these won't perform retrieval again. The generated answers and metrics will be written (in various formats) somewhere in `experiment/evaluation/<evaluation_identifier>`.

<details><summary>Show...</summary>

```sh
# evaluation (llama13b)
python main.py --mode evaluate --dataset txt --k 0 --testset human --generator llama13b --retriever bm25
python main.py --mode evaluate --dataset txt --k 0 --testset template --generator llama13b --retriever bm25

python main.py --mode evaluate --dataset txt --k 3 --testset human --generator llama13b --retriever bm25
python main.py --mode evaluate --dataset txt --k 3 --testset human --generator llama13b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 3 --testset human --generator llama13b --retriever vec
python main.py --mode evaluate --dataset txt --k 3 --testset human --generator llama13b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 3 --testset human --generator llama13b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 3 --testset human --generator llama13b --retriever ragatouille --hypothetical
python main.py --mode evaluate --dataset txt --k 3 --testset template --generator llama13b --retriever bm25
python main.py --mode evaluate --dataset txt --k 3 --testset template --generator llama13b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 3 --testset template --generator llama13b --retriever vec
python main.py --mode evaluate --dataset txt --k 3 --testset template --generator llama13b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 3 --testset template --generator llama13b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 3 --testset template --generator llama13b --retriever ragatouille --hypothetical

python main.py --mode evaluate --dataset txt --k 5 --testset human --generator llama13b --retriever bm25
python main.py --mode evaluate --dataset txt --k 5 --testset human --generator llama13b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 5 --testset human --generator llama13b --retriever vec
python main.py --mode evaluate --dataset txt --k 5 --testset human --generator llama13b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 5 --testset human --generator llama13b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 5 --testset human --generator llama13b --retriever ragatouille --hypothetical
python main.py --mode evaluate --dataset txt --k 5 --testset template --generator llama13b --retriever bm25
python main.py --mode evaluate --dataset txt --k 5 --testset template --generator llama13b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 5 --testset template --generator llama13b --retriever vec
python main.py --mode evaluate --dataset txt --k 5 --testset template --generator llama13b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 5 --testset template --generator llama13b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 5 --testset template --generator llama13b --retriever ragatouille --hypothetical

python main.py --mode evaluate --dataset txt --k 7 --testset human --generator llama13b --retriever bm25
python main.py --mode evaluate --dataset txt --k 7 --testset human --generator llama13b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 7 --testset human --generator llama13b --retriever vec
python main.py --mode evaluate --dataset txt --k 7 --testset human --generator llama13b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 7 --testset human --generator llama13b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 7 --testset human --generator llama13b --retriever ragatouille --hypothetical
python main.py --mode evaluate --dataset txt --k 7 --testset template --generator llama13b --retriever bm25
python main.py --mode evaluate --dataset txt --k 7 --testset template --generator llama13b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 7 --testset template --generator llama13b --retriever vec
python main.py --mode evaluate --dataset txt --k 7 --testset template --generator llama13b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 7 --testset template --generator llama13b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 7 --testset template --generator llama13b --retriever ragatouille --hypothetical

python main.py --mode evaluate --dataset txt --k 10 --testset human --generator llama13b --retriever bm25
python main.py --mode evaluate --dataset txt --k 10 --testset human --generator llama13b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 10 --testset human --generator llama13b --retriever vec
python main.py --mode evaluate --dataset txt --k 10 --testset human --generator llama13b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 10 --testset human --generator llama13b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 10 --testset human --generator llama13b --retriever ragatouille --hypothetical
python main.py --mode evaluate --dataset txt --k 10 --testset template --generator llama13b --retriever bm25
python main.py --mode evaluate --dataset txt --k 10 --testset template --generator llama13b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 10 --testset template --generator llama13b --retriever vec
python main.py --mode evaluate --dataset txt --k 10 --testset template --generator llama13b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 10 --testset template --generator llama13b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 10 --testset template --generator llama13b --retriever ragatouille --hypothetical


# evaluation (gemma7b)
python main.py --mode evaluate --dataset txt --k 0 --testset human --generator gemma7b --retriever bm25
python main.py --mode evaluate --dataset txt --k 0 --testset template --generator gemma7b --retriever bm25

python main.py --mode evaluate --dataset txt --k 3 --testset human --generator gemma7b --retriever bm25
python main.py --mode evaluate --dataset txt --k 3 --testset human --generator gemma7b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 3 --testset human --generator gemma7b --retriever vec
python main.py --mode evaluate --dataset txt --k 3 --testset human --generator gemma7b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 3 --testset human --generator gemma7b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 3 --testset human --generator gemma7b --retriever ragatouille --hypothetical
python main.py --mode evaluate --dataset txt --k 3 --testset template --generator gemma7b --retriever bm25
python main.py --mode evaluate --dataset txt --k 3 --testset template --generator gemma7b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 3 --testset template --generator gemma7b --retriever vec
python main.py --mode evaluate --dataset txt --k 3 --testset template --generator gemma7b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 3 --testset template --generator gemma7b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 3 --testset template --generator gemma7b --retriever ragatouille --hypothetical

python main.py --mode evaluate --dataset txt --k 5 --testset human --generator gemma7b --retriever bm25
python main.py --mode evaluate --dataset txt --k 5 --testset human --generator gemma7b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 5 --testset human --generator gemma7b --retriever vec
python main.py --mode evaluate --dataset txt --k 5 --testset human --generator gemma7b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 5 --testset human --generator gemma7b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 5 --testset human --generator gemma7b --retriever ragatouille --hypothetical
python main.py --mode evaluate --dataset txt --k 5 --testset template --generator gemma7b --retriever bm25
python main.py --mode evaluate --dataset txt --k 5 --testset template --generator gemma7b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 5 --testset template --generator gemma7b --retriever vec
python main.py --mode evaluate --dataset txt --k 5 --testset template --generator gemma7b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 5 --testset template --generator gemma7b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 5 --testset template --generator gemma7b --retriever ragatouille --hypothetical

python main.py --mode evaluate --dataset txt --k 7 --testset human --generator gemma7b --retriever bm25
python main.py --mode evaluate --dataset txt --k 7 --testset human --generator gemma7b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 7 --testset human --generator gemma7b --retriever vec
python main.py --mode evaluate --dataset txt --k 7 --testset human --generator gemma7b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 7 --testset human --generator gemma7b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 7 --testset human --generator gemma7b --retriever ragatouille --hypothetical
python main.py --mode evaluate --dataset txt --k 7 --testset template --generator gemma7b --retriever bm25
python main.py --mode evaluate --dataset txt --k 7 --testset template --generator gemma7b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 7 --testset template --generator gemma7b --retriever vec
python main.py --mode evaluate --dataset txt --k 7 --testset template --generator gemma7b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 7 --testset template --generator gemma7b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 7 --testset template --generator gemma7b --retriever ragatouille --hypothetical

python main.py --mode evaluate --dataset txt --k 10 --testset human --generator gemma7b --retriever bm25
python main.py --mode evaluate --dataset txt --k 10 --testset human --generator gemma7b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 10 --testset human --generator gemma7b --retriever vec
python main.py --mode evaluate --dataset txt --k 10 --testset human --generator gemma7b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 10 --testset human --generator gemma7b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 10 --testset human --generator gemma7b --retriever ragatouille --hypothetical
python main.py --mode evaluate --dataset txt --k 10 --testset template --generator gemma7b --retriever bm25
python main.py --mode evaluate --dataset txt --k 10 --testset template --generator gemma7b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 10 --testset template --generator gemma7b --retriever vec
python main.py --mode evaluate --dataset txt --k 10 --testset template --generator gemma7b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 10 --testset template --generator gemma7b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 10 --testset template --generator gemma7b --retriever ragatouille --hypothetical


# evaluation (mixtral8x7b)
python main.py --mode evaluate --dataset txt --k 0 --testset human --generator mixtral8x7b --retriever bm25
python main.py --mode evaluate --dataset txt --k 0 --testset template --generator mixtral8x7b --retriever bm25

python main.py --mode evaluate --dataset txt --k 3 --testset human --generator mixtral8x7b --retriever bm25
python main.py --mode evaluate --dataset txt --k 3 --testset human --generator mixtral8x7b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 3 --testset human --generator mixtral8x7b --retriever vec
python main.py --mode evaluate --dataset txt --k 3 --testset human --generator mixtral8x7b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 3 --testset human --generator mixtral8x7b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 3 --testset human --generator mixtral8x7b --retriever ragatouille --hypothetical
python main.py --mode evaluate --dataset txt --k 3 --testset template --generator mixtral8x7b --retriever bm25
python main.py --mode evaluate --dataset txt --k 3 --testset template --generator mixtral8x7b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 3 --testset template --generator mixtral8x7b --retriever vec
python main.py --mode evaluate --dataset txt --k 3 --testset template --generator mixtral8x7b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 3 --testset template --generator mixtral8x7b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 3 --testset template --generator mixtral8x7b --retriever ragatouille --hypothetical

python main.py --mode evaluate --dataset txt --k 5 --testset human --generator mixtral8x7b --retriever bm25
python main.py --mode evaluate --dataset txt --k 5 --testset human --generator mixtral8x7b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 5 --testset human --generator mixtral8x7b --retriever vec
python main.py --mode evaluate --dataset txt --k 5 --testset human --generator mixtral8x7b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 5 --testset human --generator mixtral8x7b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 5 --testset human --generator mixtral8x7b --retriever ragatouille --hypothetical
python main.py --mode evaluate --dataset txt --k 5 --testset template --generator mixtral8x7b --retriever bm25
python main.py --mode evaluate --dataset txt --k 5 --testset template --generator mixtral8x7b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 5 --testset template --generator mixtral8x7b --retriever vec
python main.py --mode evaluate --dataset txt --k 5 --testset template --generator mixtral8x7b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 5 --testset template --generator mixtral8x7b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 5 --testset template --generator mixtral8x7b --retriever ragatouille --hypothetical

python main.py --mode evaluate --dataset txt --k 7 --testset human --generator mixtral8x7b --retriever bm25
python main.py --mode evaluate --dataset txt --k 7 --testset human --generator mixtral8x7b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 7 --testset human --generator mixtral8x7b --retriever vec
python main.py --mode evaluate --dataset txt --k 7 --testset human --generator mixtral8x7b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 7 --testset human --generator mixtral8x7b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 7 --testset human --generator mixtral8x7b --retriever ragatouille --hypothetical
python main.py --mode evaluate --dataset txt --k 7 --testset template --generator mixtral8x7b --retriever bm25
python main.py --mode evaluate --dataset txt --k 7 --testset template --generator mixtral8x7b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 7 --testset template --generator mixtral8x7b --retriever vec
python main.py --mode evaluate --dataset txt --k 7 --testset template --generator mixtral8x7b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 7 --testset template --generator mixtral8x7b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 7 --testset template --generator mixtral8x7b --retriever ragatouille --hypothetical

python main.py --mode evaluate --dataset txt --k 10 --testset human --generator mixtral8x7b --retriever bm25
python main.py --mode evaluate --dataset txt --k 10 --testset human --generator mixtral8x7b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 10 --testset human --generator mixtral8x7b --retriever vec
python main.py --mode evaluate --dataset txt --k 10 --testset human --generator mixtral8x7b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 10 --testset human --generator mixtral8x7b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 10 --testset human --generator mixtral8x7b --retriever ragatouille --hypothetical
python main.py --mode evaluate --dataset txt --k 10 --testset template --generator mixtral8x7b --retriever bm25
python main.py --mode evaluate --dataset txt --k 10 --testset template --generator mixtral8x7b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 10 --testset template --generator mixtral8x7b --retriever vec
python main.py --mode evaluate --dataset txt --k 10 --testset template --generator mixtral8x7b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 10 --testset template --generator mixtral8x7b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 10 --testset template --generator mixtral8x7b --retriever ragatouille --hypothetical


# evaluation (llama70b)
python main.py --mode evaluate --dataset txt --k 0 --testset human --generator llama70b --retriever bm25
python main.py --mode evaluate --dataset txt --k 0 --testset template --generator llama70b --retriever bm25

python main.py --mode evaluate --dataset txt --k 3 --testset human --generator llama70b --retriever bm25
python main.py --mode evaluate --dataset txt --k 3 --testset human --generator llama70b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 3 --testset human --generator llama70b --retriever vec
python main.py --mode evaluate --dataset txt --k 3 --testset human --generator llama70b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 3 --testset human --generator llama70b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 3 --testset human --generator llama70b --retriever ragatouille --hypothetical
python main.py --mode evaluate --dataset txt --k 3 --testset template --generator llama70b --retriever bm25
python main.py --mode evaluate --dataset txt --k 3 --testset template --generator llama70b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 3 --testset template --generator llama70b --retriever vec
python main.py --mode evaluate --dataset txt --k 3 --testset template --generator llama70b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 3 --testset template --generator llama70b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 3 --testset template --generator llama70b --retriever ragatouille --hypothetical

python main.py --mode evaluate --dataset txt --k 5 --testset human --generator llama70b --retriever bm25
python main.py --mode evaluate --dataset txt --k 5 --testset human --generator llama70b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 5 --testset human --generator llama70b --retriever vec
python main.py --mode evaluate --dataset txt --k 5 --testset human --generator llama70b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 5 --testset human --generator llama70b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 5 --testset human --generator llama70b --retriever ragatouille --hypothetical
python main.py --mode evaluate --dataset txt --k 5 --testset template --generator llama70b --retriever bm25
python main.py --mode evaluate --dataset txt --k 5 --testset template --generator llama70b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 5 --testset template --generator llama70b --retriever vec
python main.py --mode evaluate --dataset txt --k 5 --testset template --generator llama70b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 5 --testset template --generator llama70b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 5 --testset template --generator llama70b --retriever ragatouille --hypothetical

python main.py --mode evaluate --dataset txt --k 7 --testset human --generator llama70b --retriever bm25
python main.py --mode evaluate --dataset txt --k 7 --testset human --generator llama70b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 7 --testset human --generator llama70b --retriever vec
python main.py --mode evaluate --dataset txt --k 7 --testset human --generator llama70b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 7 --testset human --generator llama70b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 7 --testset human --generator llama70b --retriever ragatouille --hypothetical
python main.py --mode evaluate --dataset txt --k 7 --testset template --generator llama70b --retriever bm25
python main.py --mode evaluate --dataset txt --k 7 --testset template --generator llama70b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 7 --testset template --generator llama70b --retriever vec
python main.py --mode evaluate --dataset txt --k 7 --testset template --generator llama70b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 7 --testset template --generator llama70b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 7 --testset template --generator llama70b --retriever ragatouille --hypothetical

python main.py --mode evaluate --dataset txt --k 10 --testset human --generator llama70b --retriever bm25
python main.py --mode evaluate --dataset txt --k 10 --testset human --generator llama70b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 10 --testset human --generator llama70b --retriever vec
python main.py --mode evaluate --dataset txt --k 10 --testset human --generator llama70b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 10 --testset human --generator llama70b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 10 --testset human --generator llama70b --retriever ragatouille --hypothetical
python main.py --mode evaluate --dataset txt --k 10 --testset template --generator llama70b --retriever bm25
python main.py --mode evaluate --dataset txt --k 10 --testset template --generator llama70b --retriever bm25 --hypothetical
python main.py --mode evaluate --dataset txt --k 10 --testset template --generator llama70b --retriever vec
python main.py --mode evaluate --dataset txt --k 10 --testset template --generator llama70b --retriever vec --hypothetical
python main.py --mode evaluate --dataset txt --k 10 --testset template --generator llama70b --retriever ragatouille
python main.py --mode evaluate --dataset txt --k 10 --testset template --generator llama70b --retriever ragatouille --hypothetical
```

</details>


## 3. (optional) Postprocess answers

Run `src/postprocess.ipynb` on the `res_acc.pkl` file produced by running `python main.py --mode evaluate`

# Experiment Notes

k = 0 are the no-context experiments. They have bm25 as the retriever, but doesn't actually use the retriever.

# Models

We identify models by llama13b, llama70b, gemma7b, mixtral8x7b. In reality, they always refer to the chat/instruction fine-tuned version.

# Tool usage

## Publicly available libraries / code

- `langchain` (MIT) as pipeline framework
- `ragatouille` (Apache-2.0) as a ColBERT wrapper
- `rank_bm25` (Apache-2.0) for non-neural retrieval baseline
- `rouge` (Apache-2.0)
- `matplotlib` (Python Software Foundation License)
- `chromadb` (Apache-2.0) 
- `stanza` (Apache-2.0) 
- `pandas` (New BSD License) for data processing 
- ATLAS's evaluation script from https://github.com/facebookresearch/atlas/blob/main/src/evaluation.py. ATLAS is CC BY-NC 4.0
- Bootstrap script by Graham https://github.com/neubig/util-scripts/blob/master/paired-bootstrap.py
- Power analysis script from https://github.com/dallascard/NLP-power-analysis (not in this repo)

Note some libraries (`sqlalchemy`) are not used by our code but something tries to import it at some point, so we listed it as a dependency. SQuAD's evaluation code is included in the repo but not used.

## Hosting

- Together.ai for hosting llama13b, gemma7b, mixtral8x7b, and llama70b
- `ollama` (MIT) to host llama13b and gemma7b (code provided in `generate.py`)

## AI tools

- GitHub Copilot is used to generate general code snippets (saving files, making directories, etc.) but not to develop the RAG system's logic

## Scraping

- `beautifulsoup4` (MIT)
- `pypdf` (BSD License)
- ScottyLab's `course-api` library (MIT) to scrape courses
- `pandoc` (GPL-2.0) for document type conversion
- Some manually scrapped results retain some html information as markdown elements. Those are done by pasting the html into the markdown editor Obsidian.
