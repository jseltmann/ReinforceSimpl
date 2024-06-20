Project for the [Reinforcement Learning for NLP](https://briemadu.github.io/rl4nlp/) course at Uni Potsdam, summer semester 2020.

This project attempts to fine-tune transformer models for language simplification using reinforcement learning. Usually, you could use a supervised approach for for this. However, there are specific rules for simple language. For example for the [Simple English Wikipedia](https://simple.wikipedia.org/wiki/Wikipedia:How_to_write_Simple_English_pages)  or for [Leichte Sprache](https://dg-ls.de/regelwerk/) in German. I tried to use these rules to create a reward function for reinforcement learning. Sadly, the transformer fine-tuning kept failing. It either returned its input again or garbled nonsense. But I still believe, that the basic idea behind this project is good.
