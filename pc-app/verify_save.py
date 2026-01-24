import numpy as np
import os
import shutil
from shared.network import ConvNetwork
from shared.models import ModelConfig
from trainer import Trainer

def test_conv_network_save_load():
    print("Testing ConvNetwork save/load...")
    net1 = ConvNetwork()
    # Modify a parameter to be sure
    net1.params['W1'][0,0,0,0] = 999.0
    
    save_path = "test_params.pkl"
    net1.save_params(save_path)
    
    net2 = ConvNetwork()
    net2.load_params(save_path)
    
    assert np.allclose(net1.params['W1'], net2.params['W1']), "W1 mismatch"
    assert net1.params['b1'].shape == net2.params['b1'].shape, "b1 shape mismatch"
    
    # Check if layer params are also updated
    assert np.allclose(net1.layers['Conv1'].W, net2.layers['Conv1'].W), "Layer Conv1 W mismatch"
    
    print("ConvNetwork save/load passed!")
    if os.path.exists(save_path):
        os.remove(save_path)

def test_trainer_init():
    print("Testing Trainer output directory creation...")
    # Mock data
    images = np.zeros((10, 3, 120, 160))
    steerings = np.zeros(10)
    throttles = np.zeros(10)
    cfg = ModelConfig(steering_min=-1, steering_max=1, throttle_min=0, throttle_max=1)
    
    trainer = Trainer(images, steerings, throttles, cfg)
    
    assert os.path.exists(trainer.output_dir), f"Output directory {trainer.output_dir} not created"
    assert os.path.isdir(trainer.output_dir), "Output dir is not a directory"
    
    print(f"Trainer created output dir: {trainer.output_dir}")
    # Cleanup
    if os.path.exists(trainer.output_dir):
        os.rmdir(trainer.output_dir)

if __name__ == "__main__":
    try:
        test_conv_network_save_load()
        test_trainer_init()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
