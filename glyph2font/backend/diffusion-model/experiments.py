
#local modules
from configs import *
from train import train_diffusion_model, train_glyphnet
from test import test_glyphnet, test_diffusion_model
from utils import encode_features
if __name__=="__main__":
    config=conf_default()
    #train_diffusion_model(config)
    test_diffusion_model(config)

    #config=conf_glyphnet()
    #train_glyphnet(config)
    #test_glyphnet(config,'data.txt')
    #encode_features(config,'/home/francisco/dataset/capitals64','/home/francisco/dataset/capitals64Features' )
