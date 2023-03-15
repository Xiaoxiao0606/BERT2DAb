from numpy import *

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience_step=50, min_mean_loss=0.2):
      
        self.patience_step = patience_step
        self.min_mean_loss = min_mean_loss
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.step_loss_list = []
    def __call__(self, step_loss):
    
        if len(self.step_loss_list) < (self.patience_step - 1):
           self.step_loss_list.append(step_loss)
        else:
            self.step_loss_list.append(step_loss)
            mean50_loss = mean(self.step_loss_list)
            if mean50_loss >= self.min_mean_loss :
                self.step_loss_list = self.step_loss_list[1:-1]
            else:
                print('INFO: Early stopping')
                self.early_stop = True

