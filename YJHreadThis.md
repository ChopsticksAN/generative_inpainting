# generative_inpainting
This is copied project of the paper "Generative Image Inpainting with Contextual Attention" in CVPR 2018

Here are some tips for YJH laotie to get start!

### Steps

- First, you shoud make the environment for this experiment.

  Get into the dir that you have just cloned and run the following command.

  ```c++
  conda env create -f KerasEnvironment.yml
  ```

  The command is to create a new environment which contains  all the pakages we need in this project.

- Then you must be eager to test the algorithm, so run the following command.

  ```c++
  pip3 install git+https://github.com/JiahuiYu/neuralgym
  ```

  This command is to install the tensorflow toolkit **neuralgym**. You just need to install it in the first time.

  Then, test it !

  ``` c++
  python3 test.py --image ./Photos/p1.jpg --checkpoint_dir ./model_logs/Places/
  ```

  The first arg **—image** ​ is to load the image dir, and the second one **—checkpoint​** is to load the model that has been trained. You can also change the images' dir or the model's.

- **Pay attention that DONT change any files as well as the test-images in the dir before you have enough confidence to use git commands.**

