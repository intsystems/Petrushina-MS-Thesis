.. class:: center

    :Title: Detection of Multimodal Hallucinations Based on Internal
Representations and Activations of Models
    :Type: Research work
    :Author: Kseniia Petrushina
    :Scientific supervisor: PhD in Computational Linguistics, Alexander Panchenko

Abstract
========

The focus of this work is on the detection of hallucinations in multimodal models in presence of unusual or commonsense-defying images. We propose methods that leverage the imperfections of large vision-language models (LVLMs), which tend to generate hallucinations when presented with images that violate everyday knowledge. Our primary method applies natural language inference (NLI) to atomic facts extracted from image descriptions, identifying internal contradictions as a signal of visual abnormality. We further explore linear probing techniques over hidden representations of LVLMs, evaluating their capacity to distinguish between realistic and unrealistic images. Both methods are evaluated on benchmarks such as WHOOPS! and WEIRD. Additionally, we consider a learning-based approach that generalizes contradiction modeling via attention pooling over fact representations. The results demonstrate that hallucination-prone behavior of LVLMs, when carefully analyzed, can serve as a valuable cue for identifying images that lack realism. This work contributes toward developing interpretable tools for multimodal hallucination detection and realism estimation.
