# Cancer Drug Sensitivity prediction from routine histology images

## Summary
Personalized drug sensitivity and treatment prediction models using histological imaging can aid in personalising cancer therapy, biomarker discovery, and drug design. However, developing these predictive models requires survival data from randomized controlled trials which can be time-consuming and expensive. In this proof-of-concept study, we demonstrate that deep learning can link histological patterns in whole slide images (WSIs) of breast cancer with drug sensitivities inferred from cancer cell lines. Specifically, we used patient-wise imputed drug sensitivities obtained from gene expression based mapping of drug effects on cancer cell lines to train a deep learning model that predicts patient sensitivity to multiple drugs from WSIs. We show that it is possible to predict the drug sensitivity profile of a cancer patient for a large number of approved and experimental drugs. Finally, we showed that the proposed approach can identify cellular and histological patterns associated with drug sensitivity profiles of individual cancer patients.

## Workflow
![workflow_github](https://github.com/engrodawood/Hist-DS/assets/13537509/6e6cbdeb-4e30-438e-b52e-f26d2152cc28)
