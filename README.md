# ClinVar
ClinVar is a public resource containing annotations about human genetic variants. These variants are (usually manually) classified by clinical laboratories on a categorical spectrum ranging from benign, likely benign, uncertain significance, likely pathogenic, and pathogenic. Variants that have conflicting classifications (from laboratory to laboratory) can cause confusion when clinicians or researchers try to interpret whether the variant has an impact on the disease of a given patient.
## Class (Conflicting or Non-Conflicting)
The objective is to predict whether a ClinVar variant will have conflicting classifications. This is presented here as a binary classification problem, where each record in the dataset is a genetic variant.

![variant](/docs/img/variant.png)

Conflicting classifications are when two of any of the following three categories are present for one variant, two submissions of one category are not considered conflicting.
1. Likely Benign or Benign
2. VUS
3. Likely Pathogenic or Pathogenic

Conflicting classification has been assigned to the CLASS column. It is a binary representation of whether or not a variant has conflicting classifications, where 0 represents consistent classifications and 1 represents conflicting classifications.
The raw variant call format (vcf) file was downloaded from: 
ftp://ftp.ncbi.nlm.nih.gov/pub/clinvar/


The raw variant format is .vcf and it cannot used for analyzing and prediction so we need to convert .vcf file to .csv file and then take use of it: :-
Online Tool used to convert this file can be found from :- http://13.59.213.190/convert
First download the zip file from :- ftp://ftp.ncbi.nlm.nih.gov/pub/clinvar/
And then extract it somewhere in your computer and then upload the .vcf file here and download the converted file from here.

![convert](/docs/img/convert_tut.png)

* CHROM :- Chromosome the variant is located on.
* POS :- Position on the chromosome the variant is located on.
* REF :- Reference Allele.
* ALT :- Alternate Allele.
* AF_ESP :- Allele frequencies from GO-ESP.
* AF_EXAC :- Allele frequencies from EXAC.
* AF_TGP :- Allele frequencies from the 1000 genomes project.
* CLASS :- The binary representation of the target class. 0 represents no conflicting submissions and 1 represents conflicting submissions.


![af_vs](/docs/img/af_vs.png)

![af_vs1](/docs/img/af_vs1.png)

![af_vs2](/docs/img/af_vs2.png)

![af_vs3](/docs/img/af_vs3.png)


From the Graph Allele Frequency from different Databases we can analyze that most of the classes in the dataset are ‘0’ means non-conflicting variant classification and some of the classes are ‘1’ means conflicting variant classification and that with ‘1’ class are very nearer to 0 to 0.15 or 0-0.2 values of Allele frequency and we can also observe that with high AF_TGP value the class will be conflicting compared with other allele frequency Database like AF_ESP, AF_EXAC.


Here in the Dataset there are many variables or features that are not contributing for the prediction model of the class because many are categorical , many observations having no values for these features and  many are correlating for example Reference Allele, Alternate Allele and many others so we drop the features and then our data prepared for training and we test with many classifiers to get the best one and this is the plot for the classifiers :-


![classifiers](/docs/img/classifiers.png)

For prediction of your test data class with default training data you just have to upload a .csv test file of some observation with unknown class in the link :- 
http://13.59.213.190/clin_var_pred

And prediction of your test data class with your training dataset you just have to upload a training data .csv file and test data .csv file in the link :- 
http://13.59.213.190/custom_clinvar

In the result you will see a confusion matrix of your training dataset and bagging classifier accuracy and result (prediction of class) of your observations in the test dataset.

![result](/docs/img/result.png)
