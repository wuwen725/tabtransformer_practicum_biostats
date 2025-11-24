\# TabTransformer Practicum – Breast Cancer Risk Stratification



This repository contains my ongoing practicum project for building

TabTransformer-based models to classify breast cancer risk using

three GEO datasets:

\- \*\*GSE164641\*\*

\- \*\*GSE95640\*\*

\- \*\*GSE240671\*\*



The project integrates gene expression data with clinical variables,

performs feature engineering, conducts differential expression analysis,

and trains TabTransformer and baseline models under cross-validation.



The repository is under active development and updated daily.



---



\## Current Project Structure



tabtransformer\_practicum\_biostats/

│

├── data/

│ ├── GSE164641/ # processed master dataframe uploaded

│ ├── GSE95640/ # processed master dataframe uploaded

│ └── GSE240671/ # processed master dataframe uploaded

│

├── notebooks/

│ ├── GSE164641/

│ │ ├── 01\_preprocessing\_and\_eda\_GSE164641.ipynb

│ │ └── 02\_model\_training\_and\_evaluation\_GSE164641.ipynb

│ ├── GSE95640/

│ │ ├── 01\_preprocessing\_and\_eda\_GSE95640.ipynb

│ │ └── 02\_model\_training\_and\_evaluation\_GSE95640.ipynb

│ └── GSE240671/

│ │ ├── 01\_preprocessing\_and\_eda\_GSE240671.ipynb

│

├── output/

│ ├── GSE164641/

│ │ ├── figures/ # AUC curves, PCA, volcano plots, etc.

│ │ └── tables/ # model performance summaries \& DEG tables

│ ├── GSE95640/

│ │ ├── figures/ # AUC curves, PCA, volcano plots, etc.

│ │ └── tables/ # model performance summaries \& DEG tables

│ └── GSE240671/

│ │ ├── figures/ # PCA, volcano plots, etc.

│

└── src/ # modularized pipeline (in progress)

├── data\_utils.py

├── feature\_utils.py

├── model\_utils.py

├── train\_utils.py

├── evaluation\_utils.py

└── plot\_utils.py





---



\## Project Progress

\- \[x] Set up project structure  

\- \[x] Added preprocessing \& EDA notebook for \*\*GSE164641\*\*  

\- \[x] Added model training \& cross-validation notebook for \*\*GSE164641\*\*  

\- \[x] Uploaded output figures and tables for \*\*GSE164641\*\*  

\- \[x] Add preprocessing \& EDA for \*\*GSE95640\*\*  

\- \[ ] Add model training for \*\*GSE95640\*\*  

\- \[ ] Upload results for \*\*GSE95640\*\*  

\- \[ ] Add preprocessing \& EDA for \*\*GSE240671\*\*  

\- \[ ] Add model training for \*\*GSE240671\*\*  

\- \[ ] Upload results for \*\*GSE240671\*\*  

\- \[ ] Finalize and integrate src/ module pipeline  

\- \[ ] Add documentation and usage instructions  



---



\## About the Project



This project aims to:

\- Integrate gene expression with clinical features

\- Perform feature engineering including DESeq2-based gene selection

\- Train TabTransformer models under stratified cross-validation

\- Compare performance against baseline models

\- Evaluate robustness across three GEO datasets



More detailed documentation will be added after all datasets'

pipelines are completed.



---



\## Datasets



\- \*\*GSE164641\*\* — Pipeline complete and uploaded  

\- \*\*GSE95640\*\* — Pipeline complete and uploaded

\- \*\*GSE240671\*\* — Processing (to be added)  



---



\## Daily Update Plan



The repository is being updated in stages to reflect a real project workflow:

1\. Preprocessing/EDA  

2\. Modeling  

3\. Results  

4\. Repeat for each dataset  

5\. Final integration  



Stay tuned for more updates!



