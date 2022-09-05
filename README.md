# individual_event_seg
This repository contains code and experimental materials for the publication "Individual variability in neural event segmentation reflects stimulus content and interpretation"

The structure of this directory is as follows:
- There is a data and code folder that are typically labeled by their corresponding figure (see below). 

**Code:**

*"event_seg"*
- This folder holds the code used to generate individual event boundaries - ["HMM_indiv_no_range_no_train.py"] - This relies on a the median number of events that were identified in our 100 ROIs. To aid in replication, we saved out a file with this number which is available in the data folder under the same name [number_of_events_dict.npy]. 
- This folder also holds code to compute the "alignment" matrices [ss_matchz.py] between pairs of subjects. This relies on the boundaries saved out from the file above. We provide matrices for each ROI in our data folder to aid others in using this code for their own analyses [matchz_matrices_concat.npy]. 

*"fig_2_4_neural_event_boundaries"*
- This folder holds code used to generate our linear models that were used to regress out the effects of motion and memory. We provide the regressors in our data folder. We also provide the outputs of our linear models that were used to generate our figures ["_linear_models" folder]. 
- This folder holds the notebooks used to generate figures 2 and 4. 
- All of the data available should make it feasible for others to do other analyses on the neural boundaries if they are interested ["data/event_seg/"]. 

*"fig_3_5_behavioral_event_boundaries"*
- This folder contains code used for our two behavioral-neural normative alignment methods that were used to generate Fig. 3. The corresponding data folders ['avg_neural_boundaries' and 'behavioral_boundaries'] that are called in this code are available.
- This folder also contains the code used to generate our linear models that were used to regress out the effects of memory from the behavioral study to generate Fig. 5. We provide the linear model outputs that were used to generate this figure in our data folder ["_linear_models_behav"] and the alignment matrices from the behavioral study ["behavioral_match_z"].
- All of the data available should make it feasible for others to do other analyses on the behavioral boundaries if they are interested. 

*"fig_6_USE_event_boundaries"*
- This folder contains code used to run the partial Mantel test used to generate the surface plots in Fig. 6A. We provide partial mantel test inputs and outputs in the corresponding data folder. 
- We also provide the code used to generate the USE matrices - "Fig_6.USE_transcript_similarity". 
- We also provide the cleaned recall transcripts in the corresponding data folder ["_all_narr_files"].
 

**experimental_files:**
- This folder contains the code used to run Experiment 2 - auxilliary behavioral experiment and the corresponding code used to administer the study (.html file) using JsPsych. Note JsPsych 7 was used to run this experiment. There is also a .txt file providing more detail on what questions were used.
