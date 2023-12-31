# individual_event_seg
This repository contains code and experimental materials for the publication "Individual differences in neural event segmentation of continuous experiences"

<p align="center">
  <a href="[https://www.biorxiv.org/content/10.1101/2022.09.09.507003v2](https://watermark-silverchair-com.dartmouth.idm.oclc.org/bhad106.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA3IwggNuBgkqhkiG9w0BBwagggNfMIIDWwIBADCCA1QGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMJAeMojp7npGAMv5GAgEQgIIDJWUGZOZ8PMP9bD2z_98JLnpv-xHiFhg0TSXX0krn5TlfDZym60hmrYZJ6fxNEs-YLvETzeWUdnXTdzYs4y7l_jj_duMTJpoK_hsx_ig5-EtK8dF-2Fj-t6vyAxSTBpP7_KX9XYnORuNIB0XqWO01U9D-kDI5SM8xX88tkTt4_6qcxQXK4UYfyIPTsI6UzKrD9J43ltL76DNz7RZ0P0vOFKaCCigCv8jQEhz6KhCnvAiUaOJIFWRh60bPdrH2qhzizRkUFFi0xOIiimRIxBZA_b2n7DOWtBtJbbEhuSB8Wp5N7zcC4YTdgC1OsdH8Y7G8qMTcDf6ei4h4ln-g7Gn2huJzLIimy-0aJWXpugGg8FpP0hlCOqQyeH4fzcc23Tof9Z7YSh7ZFUKe3vy4AVHdYYsutV9NfmJaYXZge1gepJYf3eMZ5o5ymt2H8njq69h5aPGt-Cum_pvI4_mneL-LmqMeCPl6M7tZ6qHj458jMOdCXjADR3d7ACc0CTvsmPOIXWlDKP_4vtsDKil79mZoPfJc2g5RmfkMjbkPS-qdMf8czVpH3lAjsG7qzsHYrNu-M9VwI80fI0kwszMnAou5AxtXv8TPwxzDiGSKMI6aVM3HUY48r608-Uc5Zeww4E31UfOcvG4cKHh-ENjZGzerGEzVE61NbuHGeuYqc4He-eNoi9byqTephpQu6B9lGtyt_prmu2LKGaWkecct4pETOjYc5v8nbYWy5-vyPzD6mrha-_xgbDctWg9wZO4GMAVJsoDTI7NqDQb0FMtQyAOYpWsHqYA1bZbMOUr52keSYnSoIGcMI-mtsYM1SJjRvbvH-BQcKrych-AGinaYnfIXm3977z3IqXWG1p9UvEwBlRbMXK_tVdcSiUkIIC8lK2HZpJjfblzEbQJf_ZE3S8h5SQ5vRC0Ss4CeIrScAMsRO9nkvjFd4SacaQbWs9AIu-FPhzTR01GckD4vXNvq_s5vbCM9y5-LnFTx7gqfpr9JSxTxVYj5mrIN2iYY4HZoZ1RYwT2hIaume1zWzhraEm0r7kqSVXLCxfL2pLM1D1lrcy1C7WaGtYI)">
<img src="https://raw.githubusercontent.com/csavasegal/csavasegal.github.io/main/img/paper.png" alt="Paper" width="100" height="100">

  </a>
</p>

The structure of this directory is as follows:
- There is a data and code folder that are typically labeled by their corresponding figure (see below). 

**Code:**

*"event_seg"*
- This folder holds the code used to generate individual event boundaries - ["HMM_indiv_no_range_no_train.py"] - This relies on a the median number of events that were identified in our 100 ROIs. To aid in replication, we saved out a file with this number which is available in the data folder under the same name [number_of_events_dict.npy]. 
- This folder also holds code to compute the "alignment" matrices [ss_matchz.py] between pairs of subjects. This relies on the boundaries saved out from the file above. We provide matrices for each ROI in our data folder to aid others in using this code for their own analyses [matchz_matrices_concat.npy]. 

*"fig_1_2_neural_event_boundaries"*
- This folder holds code used to generate our linear models that were used to regress out the effects of motion and memory. We provide the regressors in our data folder. We also provide the outputs of our linear models that were used to generate our figures ["_linear_models" folder]. 
- This folder holds the notebooks used to generate figures 1 and 2. 
- All of the data available should make it feasible for others to do other analyses on the neural boundaries if they are interested ["data/event_seg/"]. 

*"fig_2_3_behavioral_event_boundaries"*
- This folder contains code used for our two behavioral-neural normative alignment methods that were used to generate Fig. 3. The corresponding data folders ['avg_neural_boundaries' and 'behavioral_boundaries'] that are called in this code are available.
- This folder also contains the code used to generate our linear models that were used to regress out the effects of memory from the behavioral study to generate Fig. 2. We provide the linear model outputs that were used to generate this figure in our data folder ["_linear_models_behav"] and the alignment matrices from the behavioral study ["behavioral_match_z"].
- All of the data available should make it feasible for others to do other analyses on the behavioral boundaries if they are interested. 

*"fig_3_USE_event_boundaries"*
- This folder contains code used to run the partial Mantel test used to generate the surface plots in Fig. 3A. We provide partial mantel test inputs and outputs in the corresponding data folder. 
- We also provide the code used to generate the USE matrices - "Fig_3a.USE_transcript_similarity". 
- We also provide the cleaned recall transcripts in the corresponding data folder ["_all_narr_files"].
 

**experimental_files:**
- This folder contains the code used to run Experiment 2 - auxilliary behavioral experiment and the corresponding code used to administer the study (.html file) using JsPsych. Note JsPsych 7 was used to run this experiment. There is also a .txt file providing more detail on what questions were used.
