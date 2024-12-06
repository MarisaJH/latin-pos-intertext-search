# Part of speech enhanced search for Latin intertexts

This codebase is forked from [NAACL-HLT-2021-Latin-Intertextuality](https://github.com/QuantitativeCriticismLab/NAACL-HLT-2021-Latin-Intertextuality), based on the paper Burns, Brofos, Li, Chaudhuri, and Dexter 2021, ["Profiling of Intertextuality in Latin Literature Using Word Embeddings"](https://www.aclweb.org/anthology/2021.naacl-main.389/) in *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*.

I have added two features: 
  1) ability to use contextual embeddings rather than static embeddings (I use [LaBERTa](https://huggingface.co/bowphs/LaBerta/tree/main))
  2) ability to filter retrieved intertexts by POS and morpholgical constraints (I used [LatinCy](https://github.com/diyclassics/llatincy) for tagging)
