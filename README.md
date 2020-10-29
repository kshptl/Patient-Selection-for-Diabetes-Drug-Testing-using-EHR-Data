# Regression Model to Predict Patient Estimated Hospitalization Time
Drugs that are ready for Phase III clinical trial testing undergo administering and screening. For example, a very unique and sensitive diabetes drug requires administering and screening the drug over at least 5-7 days of time in the hospital with frequent monitoring/testing and patient medication adherence training with a mobile application. A predictive model was built that can identify which type of patients the drug should be tested on. Target patients were people that were likely to be in the hospital for this duration of time and would not incur significant additional costs for administering this drug to the patient and monitoring.  

In order to achieve this goal a regression model was built that could predict the estimated hospitalization time for a patient and use this to select/filter patients for the study.

**Expected Hospitalization Time Regression Model:** Utilizing a synthetic dataset(denormalized at the line level augmentation) built off of the UCI Diabetes readmission dataset, a regression model was built that predicts the expected days of hospitalization time and then convert this to a binary prediction of whether to include or exclude that patient from the clinical trial.

This project demonstrated the importance of building the right data representation at the encounter level, with appropriate filtering and preprocessing/feature engineering of key medical code sets. This model was also analyzed and interpreted for biases across key demographic groups. 

### Dataset
Due to healthcare PHI regulations (HIPAA, HITECH), there are limited number of publicly available datasets and some datasets require training and approval. Therefore, dataset is limited in its representation of some key features such as diagnosis codes which are usually an unordered list in 835s/837s (the HL7 standard interchange formats used for claims and remits).

- https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008
