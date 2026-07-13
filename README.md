This is the Python code used to generate the results described in the article "Corticofugal gated recurrency captures auditory cortical responses".

The data is first converted to a format suitable for processing in Python in each _prebuild.py_ script in the respective dataset folder ('NS3' and 'NS3_PEG' correspond to 'NAT4-A1' and 'NAT4-PEG' in the paper). Pre-processing steps as detailed in the paper are contained in the _build_datasets.py_ file. The main script is _pipeline.py_, used for the initial model fit. The _test_pipeline_norefit.py_ script is for model testing. The further analyses detailed in the paper are contained in the _final_analysis.py_, _memory_retention.py_, and _rcLSTM_control_analysis.py_ scripts. The _Temporal_prediction_, _SSA_, and _FRMs_ folders contain scripts for the analyses indicated in their name.

Some functions, relating to pre-processing and performance calculation, were adapted from [benlib-py](https://github.com/ben-willmore/benlib-py). The performance of models on the sequential MNIST task was evaluated using [miltonllera](https://github.com/miltonllera/pytorch-subLSTM)'s implementation.

***

The data can be found here: [NS3 and NS3_PEG](https://zenodo.org/records/7796574); [NS2](https://zenodo.org/records/3445557). To run the scripts, download the data in its original format to the corresponding subfolders within the _Data_ folder. In the main training script, choose the dataset and model, as well as the model hyperparameters. Make sure that model fitting results (performance, model parameters, and meta-info) are being saved. For testing, choose the saved result file of the fitted model being tested.
