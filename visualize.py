import wandb
import traceback

# HOW TO USE
# in run_train.py, add the decorators like this:
#
#   @make_wandb
#   def make(config):
#       ...
#   
#   @train_log_wandb
#   def train_log(loss, example_ct, epoch):
#       ...
#
#   @save_wandb
#   def save(model, path): :
#       ...

# Decorator to wrap the make(config) function
def make_wandb(make_func):
    def wrapper_make(*args, **kwargs):
        ret = make_func(*args, **kwargs)
        # === try to log the config ===
        try:
            config = vars(args[0])
            wandb.init(config=config)
        except Exception:
            traceback.print_exc()
        return ret
    return wrapper_make

# Decorator to wrap the train_log(loss, example_ct, epoch) function
def train_log_wandb(train_log_func):    
    def wrapper_train_log(*args, **kwargs):
        ret = train_log_func(*args, **kwargs)
        # === try to log the config ===
        try:
            loss = args[0]
            wandb.log({"loss": loss})
        except Exception:
            traceback.print_exc()  
        return ret
    return wrapper_train_log

# Decorator to wrap the save(model, path) function
def save_wandb(save_func):
    def wrapper_save(*args, **kwargs):
        ret = save_func(*args, **kwargs)
        # === get and save the model ===
        try:
            model, path = args
            name = path.replace('/', '-')
            art = wandb.Artifact(name=f'model_{name}', type='model state dict')
            art.add_file(path)
            wandb.log_artifact(art)
        except Exception:
            traceback.print_exc()
        return ret
    return wrapper_save