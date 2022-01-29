# Two-Stage Classification Model for Scanning Electron Microscope Images in Materials Science
 Repository For our Paper Two-Stage Classification Model for Scanning Electron Microscope Images in Materials Science
 
 ## Training The Model:
    Steps :
        1. Download The Dataset From ___
        2. Extract the Zip Folder in /Dataset
        3. Run Train.py
             ```
             python train.py d_opt
             ```
             When d_opt =0 it would Train on the SEM Dataset
             When d_opt =1 it would Train on the Particle Datset.
    
    <i> The Models Thus Generated Would Be Stored in the Models Folder <\br> The Plots Generated would be stored in Plots Folder. <\i>
 ## Options.py
    Options.py Contains the training options for both SEM and Particle Dataset . To change any parameter in training process update the options.py.
 
