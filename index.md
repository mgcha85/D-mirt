# D-miRT: A two-stream convolutional neural network for mi-croRNA transcription start site feature integration and identi-fication
#### Mingyu Cha<sup>1</sup>, Amlan Talukder<sup>1</sup>, Clayton Barham<sup>1</sup>, Xiaoman Li<sup>2*</sup>, Haiyan Hu<sup>1*</sup>

## ABSTRACT

MicroRNAs (miRNAs) play important roles in post-transcriptional gene regulation and phenotype development. Under-standing the regulation of miRNA genes is critical to understanding gene regulation. One of the challenges to studying miRNA gene regulation is the lack of condition-specific annotation of miRNA transcription start sites (TSSs). Unlike pro-tein-coding genes, miRNA TSSs can be tens of thousands of nucleotides away from the precursor miRNAs and they are hard to be detected by conventional RNA-Seq experiments. A number of studies have been attempted to computation-ally predict miRNA TSSs. However, high-resolution condition-specific miRNA TSS prediction remains a challenging problem.  Recently, deep learning models have been successfully applied to various bioinformatics problems but have not been effectively created for condition-specific miRNA TSS prediction. Here we created a two-stream deep learning model called D-miRT for computational prediction of condition-specific miRNA TSSs. D-miRT is a natural fit for the integration of low-resolution gene transcription activation markers such as DNase-Seq and histone modification data and high-resolution sequence features. We trained the D-miRT model by integrating genome-scale CAGE experiments and transcription activation marker data across multiple cell lines. Compared with alternative computational models on different sets of training data, D-miRT outperformed all baseline models and demonstrated high accuracy for condition-specific miRNA TSS prediction tasks. Comparing with the most recent approaches on cell-specific miRNA TSS identifi-cation using cell lines that were unseen to the model training processes, D-miRT also showed superior performance.

## Download
The source code of DmiRT is available [here] (http://hulab.ucf.edu/research/projects/DmiRT/DmiRT.zip).  

## Manual
The manual for running the program is available [here] (http://hulab.ucf.edu/research/projects/DmiRT/DmiRT.txt).  


## Authors
Mingyu Cha<sup>1</sup>, Amlan Talukder<sup>1</sup>, Clayton Barham<sup>1</sup>, Xiaoman Li<sup>2*</sup>, Haiyan Hu<sup>1*</sup>
Department of Computer Science, University Of Central Florida, Orlando, FL 32826, USA.




