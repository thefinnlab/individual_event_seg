# individual_event_seg
This repository contains code and experimental materials for the publication "name TBD"

The structure of this directory is as follows:
- There is a data and code folder that are typically labeled by their corresponding figure (see below). 

*Code:*
*** "event_seg" ***
- This folder holds the code used to generate individual event boundaries - "FILE NAME HERE"
- This relies on a the median number of events that were identified in our 100 ROIs. To aid in replication, we saved out a file with this number which is available in data. 
- This folder also holds code to compute the "alignment" matrices [ss_matchz.py] between pairs of subjects. This relies on the boundaries saved out from the file above. We provide a sample ROI's matrix in our data folder to aid others in using this code for their own analyses. 

*** "fig_2_4_neural_event_boundaries" ***
- This folder holds code used to generate our linear models that were used to regress out the effects of motion and memory. We provide the regressors in our data folder. We also provide the outputs of our linear models that were used to generate our figures. 
- This folder holds the notebooks used to generate figures 2 and 4. 

*** "fig_3_5_behavioral_event_boundaries" ***
- This folder contains code used for our two behavioral-neural normative alignment methods ["FILE NAME HERE"] that were used to generate Fig. 3
- This folder also contains the code used to generate our linear models that were used to regress out the effects of memory from the behavioral study to generate Fig. 5. We provide both the behavioral data matrices and the regressor matrix in our data folder. 

*** "fig_6_USE_event_boundaries" ***
- This folder contains code used to prepare for and run the partial Mantel test used to generate the neural images in Fig. 6. 
- The USE matrices for each movie are also found in the data folder as is the code used to generate these matrices. 
- We also provide the cleaned recall transcripts on OpenNeuro [CHECK THIS - maybe host on Github]
