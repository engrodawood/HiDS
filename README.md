# Cancer Drug Sensitivity prediction from routine histology images

## Summary
Personalized drug sensitivity and treatment prediction models using histological imaging can aid in personalising cancer therapy, biomarker discovery, and drug design. However, developing these predictive models requires survival data from randomized controlled trials which can be time-consuming and expensive. In this proof-of-concept study, we demonstrate that deep learning can link histological patterns in whole slide images (WSIs) of breast cancer with drug sensitivities inferred from cancer cell lines. Specifically, we used patient-wise imputed drug sensitivities obtained from gene expression based mapping of drug effects on cancer cell lines to train a deep learning model that predicts patient sensitivity to multiple drugs from WSIs. We show that it is possible to predict the drug sensitivity profile of a cancer patient for a large number of approved and experimental drugs. Finally, we showed that the proposed approach can identify cellular and histological patterns associated with drug sensitivity profiles of individual cancer patients.

## Workflow
![workflow_github](https://github.com/engrodawood/Hist-DS/assets/13537509/6e6cbdeb-4e30-438e-b52e-f26d2152cc28)

## Training and Evaluation

Workspace directory contain necessary script for constructing graph and training the proposed SlideGraph<sup>∞</sup>. 

Step1: Download TCGA BRCA Diagnostic slides from <a href='https://docs.gdc.cancer.gov/Data_Portal/Users_Guide/Repository/'>GCD data portal</a>

Step2: Download tissue segmentation mask from this <a href = "https://drive.google.com/file/d/1nvGyMm33gl-iYlVEziM_RjpL1c61ApXv/view?usp=sharing"> Link</a>.

Step3: Generate patches of each Whole slide image by running
  ```python patches_extraction.py```

Step4: Extract ShuffleNet representation from each of the WSI patch by running
   ```python deep_features.py```

Step5: Construct WSI-Graph by running
   ```python graph_construction.py```

Step6: Training the Graph Neural Network by running
   ```python main.py```

## License
The source code of SlideGraph<sup>∞</sup> is released under MIT-CC-Non-Commercial license.
