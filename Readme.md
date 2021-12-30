## Weather Image Classification

- Dataset can be downloaded from [here](https://www.kaggle.com/jehanbhathena/tire-texture-image-recognition)

- Rename `___.zip` to `data.zip`

- Run `unzip.py [args]` once dataset is downloaded. Adjust folder structure to `data/training_data` and `data/testing_data`

- Run `np_make.py [args]` to convert entire dataset into a `np.uint8` array

- Run `main.py [args]` to run the program

<br>

### **CLI Arguments**

<br>

<pre>
1. --path | -p            : Path where numpy arrays are stored after running np_make (Default: data)
2. --seed | -s            : Seed Value (Default: 0)
3. --backbone | -bb       : Model Backbone (Default: mobilenet) (Available: vgg, resnet, densenet, mobilenet)
4. --mode | -m            : Model Train Mode (Default: full) (Available: full, semi, final)
5. --epochs | -e          : Epochs (Default: 10)
6. --early-stopping | -es : Early Stopping (Default: 5)
7. --batch-size | -bs     : Batch Size (Default: 64)
8. --learning-rate | -lr  : Learning Rate (Default: 1e-3)
9. --weight-decay | -wd   : Weight Decay (Default: 0.0)
10. --size                : Image Size (Use same value as when running np_make.py)
11. --augment             : Flag to augment the training images
12. --patience-eps        : Plateau Scheduler arguments (Expects as 'patience' 'eps')
13. --test                : Flag to enter testing mode (Must be followed up by an image name as 'image_name.ext')
14. --kaggle | -k         : Enables scripts to work within a kaggle notebookvls
</pre>