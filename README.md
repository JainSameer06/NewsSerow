# NewsSerow

## Few-Shot Environmental Conservation Media Monitoring for Low-Resource Languages (AAAI 2024 - AI for Social Impact Track)

<p align="center">
  <a href="https://aaai.org/aaai-conference/"><img src="https://img.shields.io/badge/AAAI-2024-blue"></a>
  <a href="https://arxiv.org/abs/2402.11818"><img src="https://img.shields.io/badge/Paper-PDF-red"></a>
  <a href="https://github.com/JainSameer06/NewsSerow/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green">
  </a>
</p>

Environmental conservation organizations routinely monitor news content on conservation in protected areas to maintain situational awareness of developments that can have an environmental impact. Existing automated media monitoring systems require large amounts of data labeled by domain experts, which is only feasible at scale for high-resource languages like English. However, such tools are most needed in the global south where news of interest is mainly in local low-resource languages, and far fewer experts are available to annotate datasets sustainably. We propose NewsSerow, a method to automatically recognize environmental conservation content in low-resource languages. NewsSerow is a pipeline of summarization, in-context few-shot classification, and self-reflection using large language models (LLMs). Using at most 10 demonstration example news articles in Nepali, NewsSerow significantly outperforms other few-shot methods and achieves comparable performance with models fully fine-tuned using thousands of examples. The World Wide Fund for Nature (WWF) has deployed NewsSerow for media monitoring in Nepal, significantly reducing their operational burden, and ensuring that AI tools for conservation actually reach the communities that need them the most. NewsSerow has also been deployed for countries with other languages like Colombia.


## Installation

We recommend creating a conda environment for this setup. Steps to install conda can be found [here](https://docs.anaconda.com/free/miniconda/). Once conda has been installed, create an environment using the requirements.txt:

```conda create --name <env> --file requirements.txt```

## Data

The data used for the experiments in the paper is given in the `data` directory. It consists of Nepali and Colombian news articles labeled for relevance to environmental conservation by domain experts from WWF.

It is possible to use our implementation to classify articles in any other language, assuming at least 10 labeled examples are available in the same language. The example and test files should be named `incontext_<language>.csv` and `<language>_test.csv`. The csv files should contain the following columns:
- `title`: The title of the news article
- `content`: The text of the news article
- `conservation_label`: The true conservation label of the article. Required only for the examples in the `incontext_<language>.csv` file.

## Usage

The results given in the paper can be replicated by running `bash run_gptclassifier.sh` from within the `src` directory. This runs the pipeline for both Nepali and Colombian data, and creates a `results` directory with the results.

Our implementation can easily be adapted to classify articles in any other language. In addition to making the changes described in the Data section above, one needs to replace _nepali_ and _colombian_ in line 7 of `run_gptclassifier.sh` with the language of interest.

Our paper describes how each of our prompt engineering features and modules (e.g. chain-of-thought and zero-shot summarization) help performance. However, the usage of _reflection_ depends on preference and situational requirements. Concretely, when reflection is used, the pipeline passes _positively_ predicted examples through an additional screening prompt. This helps significantly reduce the number of false positives by flipping the predicted label for articles that are only tenuously related to conservation(Sections 3.4 and 4.5). While this helps boost precision, it causes a drop in recall.