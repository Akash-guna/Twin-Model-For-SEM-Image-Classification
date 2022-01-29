def options(d_opt):
    '''
   Input :
    d_opt --> Mentions if dataset used for training is SEM or Particle
   
   Output:
    SEM_PATH --> Path to Normal Image Dataset
    LBP_PATH --> Path to LBP Image Dataset
    batch_size --> Batch size
    categories --> Categories present in dataset
    train_len --> Length of training Set i.e Number of Images
    val_len --> Length of Validation Dataset
    normal_epochs --> Epochs of Normal Model
    lbp_epochs --> Epochs of LBP Model
    multi_epochs --> Epochs of Multi Model
   Use:
        Defines the Options for Various Dataset training options.
    '''
    if int(d_opt) == 0:
        print('Training on SEM Dataset')
        SEM_PATH = 'Dataset/SEM'
        LBP_PATH = 'Dataset/SEM_LBP'
        batch_size = 16
        categories =10
        train_len =16717
        val_len = 1853
        normal_epochs = 40
        lbp_epochs = 60
        multi_epochs = 5
    else:
        print('Training on Particles Dataset')
        SEM_PATH = 'Dataset/Particle'
        LBP_PATH = 'Dataset/Particle_LBP'
        batch_size = 16
        categories =8
        train_len =1840
        val_len = 200
        normal_epochs = 60
        lbp_epochs = 100
        multi_epochs = 5
    return [SEM_PATH,LBP_PATH,batch_size,categories,train_len,val_len,normal_epochs,lbp_epochs,multi_epochs]