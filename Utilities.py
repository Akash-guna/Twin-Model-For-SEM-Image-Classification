import matplotlib.pyplot as plt

def plot_accuracy(history,save_dest):
    '''
    Inputs :
        History   --> History Object from Keras Fit For Plotting
        save_dest --> Destination for saving the plots
    Outputs :
        BOOLEAN --> Success or Failure
    Use  :
        Plots the Accuracy Vs Epoch Graph
    '''
    try:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Accuracy vs Epoch')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(save_dest)
        return True
    except:
        return False

def plot_loss(history,save_dest):
     '''
    Inputs :
        History   --> History Object from Keras Fit For Plotting
        save_dest --> Destination for saving the plots
    Outputs :
        BOOLEAN --> Success or Failure
    Use  :
        Plots the Loss Vs Epoch Graph
    '''
     try:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Loss vs Epoch')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(save_dest)
        return True
     except:
        return False

def plot_history(history,folder,kind):
    '''
    Inputs :
        History   --> History Object from Keras Fit For Plotting
        folder --> SEM / Particle Dataset
        kind   --> Normal / LBP/ Multi Model
    Outputs :
        None
    Use  :
        Calls All Plotting Methods and Handles Tracebacks.
    '''
     # Save Accuracy Plot
    save_path = f'Plots/{folder}_Accuracy_{kind}.jpg'
    v = plot_accuracy(history,save_path)
    if v== True:
        print(f'Saved Accuracy Plot at {save_path}')
    else:
        print('Failed To save Accuracy Plot')

    # Save Loss Plot
    save_path = f'Plots/{folder}_Loss_{kind}.jpg'
    v = plot_accuracy(history,save_path)
    if v== True:
        print(f'Saved Loss Plot at {save_path}')
    else:
        print('Failed To save Loss Plot')


