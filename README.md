<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />

<div align="center">
  <a href="https://github.com/alaaNfissi/CNN-n-GRU-for-speech-emotion-recognition">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">From Unaltered Raw Waveform to Emotion: Synergizing Convolutional and Gated Recurrent Networks for Holistic Speech Emotion Analysis</h3>

  <p align="center">
    This paper has been accepted for publication in Applied Intelligence.
    <br />
   </p>
   <!-- <a href="https://github.com/alaaNfissi/CNN-n-GRU-for-speech-emotion-recognition"><strong>Explore the docs »</strong></a> -->
</div>
   

  
<div align="center">

[![view - Documentation](https://img.shields.io/badge/view-Documentation-blue?style=for-the-badge)](https://github.com/alaaNfissi/CNN-n-GRU-for-speech-emotion-recognition/#readme "Go to project documentation")

</div>  


<div align="center">
    <p align="center">
    ·
    <a href="https://github.com/alaaNfissi/CNN-n-GRU-for-speech-emotion-recognition/issues">Report Bug</a>
    ·
    <a href="https://github.com/alaaNfissi/CNN-n-GRU-for-speech-emotion-recognition/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#abstract">Abstract</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#getting-the-code">Getting the code</a></li>
        <li><a href="#dependencies">Dependencies</a></li>
        <li><a href="#using-the-code">Using the code</a></li>
      </ul>
    </li>
    <li>
      <a href="#results">Results</a>
      <ul>
        <li><a href="#on-iemocap-dataset">On IEMOCAP dataset</a></li>
        <li><a href="#on-tess-dataset">On TESS dataset</a></li>
        <li><a href="#on-ravdess-dataset">On RAVDESS dataset</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<div align="center">
  
![ser-approaches][ser-approaches]
  
*Workflows in Speech Emotion Recognition*
  
</div> 

<!-- ABSTRACT -->
## Abstract

<p align="justify"> Recognizing emotions from speech remains a challenging task due to the variability of vocal expression and the information loss associated with conventional feature extraction methods. This study introduces CNN-n-GRU, an end-to-end deep learning architecture developed to perform speech emotion recognition directly from raw waveform data. The model combines an n-layer convolutional neural network for extracting hierarchical local acoustic features with an n-layer gated recurrent unit designed to model long-term temporal dependencies in speech. The convolutional component acts as a feature extractor, generating progressively abstract representations of the input signal. These features are then passed to the gated recurrent layers, which capture sequential information and selectively retain emotionally relevant cues. By eliminating the need for handcrafted features or spectrogram transformations, the model preserves narrow-band emotional information and can efficiently handle speech signals of varying durations without explicit segmentation. The design draws conceptual motivation from auditory perception: the initial convolutional layers emulate Cochlear frequency selectivity, while the recurrent layers mirror the brain's ability to focus on salient acoustic patterns over time. The proposed architecture is evaluated on three benchmark datasets. On TESS, the model achieves 99.2% accuracy and 99.0% F1-score, on the IEMOCAP dataset, it reaches 81.3% accuracy and 80.9% F1-score, and on the RAVDESS dataset, it attains an accuracy of 86.6% and an F1-score of 86.7%. These results represent improvements over state-of-the-art methods, with statistical significance and additional analysis supporting the consistency and robustness of the performance. Although effective in controlled data sets, the generalizability of the model to spontaneous emotional speech and speaker variability remains an open area for exploration. Future work will focus on expanding its applicability to more diverse, real-world acoustic environments. </p>
<div align="center">
  
![model-architecture][model-architecture]
  
*Proposed model architecture*
  
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With
* ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
* ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
* ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
* ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
* ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
* ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started
<p align="justify">
To begin our experiments, we first ensured that our signal has a sampling rate of 16 KHz and is mono-channel in order to standardise our experimental data format.
Each dataset is segmented as follows: 80\% for training, 10\% for validation, and 10\% for testing based on stratified random sampling which entails categorising the whole population into homogenous groupings known as strata. Random samples are then drawn from each stratum unlike basic random sampling which considers all members of a population as equal. With an equal possibility of being sampled, it allows us to generate a sample population that best represents the total population being studied as it is used to emphasise distinctions across groups in a population. A Grid search is then used to find the appropriate hyperparameters. Some hyperparameter optimization approaches are known as "scheduling algorithms". These Trial Schedulers have the authority to terminate troublesome trials early, halt trials, clone trials, and alter trial hyperparameters while they are still running. Thus, the Asynchronous Successive Halving algorithm (ASHA) was picked because of its high performance.
  
We examined four model architectures: CNN-3-GRU, CNN-5-GRU, CNN-11-GRU, and CNN-18-GRU. Each model is run for 100 epochs until it converges using Adam. As we are not using any pretrained model, the weights of each model are started from scratch. The receptive field of our first CNN layer is equal to <em>160</em> which corresponds to <em>(sampling rate / 100)</em> in our case to cover a <em>10-millisecond</em> time span, to be comparable to the window size for many MFCC computations since we transformed all our data to <em>16 KHz</em> representation. All source code used to generate the results and figures in the paper are in
the project repository. This unified implementation supports TESS, IEMOCAP, and RAVDESS datasets through a central configuration file. See the details below for a full description.  
</p>

### Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/alaaNfissi/CNN-n-GRU-for-speech-emotion-recognition.git

or [download a zip archive](https://github.com/alaaNfissi/CNN-n-GRU-for-speech-emotion-recognition/archive/refs/heads/main.zip).

### Dependencies

<p align="center">

You'll need a working Python environment to run the code.
The recommended way to set up your environment is through the
[Anaconda Python distribution](https://www.anaconda.com/download/) which
provides the `conda` package manager.
Anaconda can be installed in your user directory and does not interfere with
the system Python installation.
The required dependencies are specified in the file `requirements.txt`.
We use `conda` virtual environments to manage the project dependencies in
isolation.
Thus, you can install our dependencies without causing conflicts with your
setup (even with different Python versions).
Run the following command to create an `ser-env` environment to create a separate environment:

    conda create --name ser-env

Activate the environment, this will enable the it for your current terminal session. Any subsequent commands will use software that is installed in the environment:

    conda activate ser-env

Use Pip to install packages to Anaconda Environment:

    conda install pip

Install all required dependencies in it:

    pip install -r requirements.txt

  
</p>

### Using the code

<p align="center">  
  
1. First, you need to download the datasets:
  * [TESS official website](https://tspace.library.utoronto.ca/handle/1807/24487)
  * [IEMOCAP official website](https://sail.usc.edu/iemocap/)
  * [RAVDESS official website](https://zenodo.org/record/1188976)
  
2. Update the dataset paths in the configuration file:

Open `config.py` and set the paths to your downloaded datasets:

```python
# Dataset selection - change this to switch between datasets
# Options: "tess", "iemocap", or "ravdess"
DATASET = "tess"

# Directory paths - update these to your local paths
# Recommended structure is to place datasets in a 'datasets' folder
TESS_DATA_FOLDER = "../datasets/TESS"
IEMOCAP_DATA_FOLDER = "../datasets/IEMOCAP"
RAVDESS_DATA_FOLDER = "../datasets/RAVDESS"
```

3. Make sure your ser-env environment is activated:

```bash
conda activate ser-env
```

4. Run the code with default settings:

```bash
# Navigate to the project directory
cd CNN-n-GRU-for-speech-emotion-recognition

# Run the main script
python main.py
```

5. For custom options:

```bash
# Train a specific model using GPU
python -m main --model cnn18gru --train --test

# Evaluate a pre-trained model
python -m main --model cnn18gru --test --load_checkpoint path/to/checkpoint.pt

# Use dropout regularization for better generalization
python -m main --model cnn18gru --train --test --dropout 0.2

# Perform hyperparameter grid search to find optimal configuration
python -m main --model cnn3gru --grid_search

# See all available options
python -m main --help
```

6. Use different datasets:

```bash
# Use the IEMOCAP dataset
python -m main --dataset iemocap --train --test

# Use the RAVDESS dataset
python -m main --dataset ravdess --train --test
```

7. Advanced training options:

```bash
# Train with custom hyperparameters
python -m main --model cnn18gru --train --test --lr 0.001 --hidden_dim 128 --num_layers 2 --dropout 0.3 --batch_size 64

# Train for a specific number of epochs
python -m main --model cnn18gru --train --test --epochs 50
```

8. Hyperparameter Grid Search:

The project includes a comprehensive grid search functionality to automatically find the best hyperparameters for your model and dataset. This leverages Ray Tune to efficiently search across a large hyperparameter space.

```bash
# Run grid search for CNN3GRU model
python -m main --model cnn3gru --grid_search

# Run grid search for CNN18GRU model
python -m main --model cnn18gru --grid_search
```

The grid search explores relevant hyperparameters including learning rates, weight decay, batch sizes, model architecture dimensions, dropout rates, optimizers, and schedulers. This helps identify the optimal configuration for your specific dataset and use case, significantly improving model performance without manual tuning.

The best model configuration will be saved to the experiments directory and can be loaded for later use.

```bash
# Load and test the best model found by grid search
python -m main --model cnn3gru --test --load_checkpoint experiments/tess/cnn3gru/best_model.pth
```

For larger datasets or more complex models, you can modify the `grid_search` function in `main.py` to customize the hyperparameter search.
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Results
<p align="center">  
  
We implemented the proposed architecture CNN-n-GRU in four versions, with n = 3, 5, 11, and 18.
  
</p>

### On IEMOCAP dataset
<p align="center">  
  
Amoung our model's four versions performance, the best architecture of our model is CNN-18-GRU as it achieves the highest accuracy and F1-score, 
where it reaches 81.3% accuracy and 80.9% F1-score on the IEMOCAP dataset which is better compared to the state of-the-art methods.
The CNN-18-GRU training and validation accuracy over epochs figure shows the evolution of training and validation accuracy of the CNN-18-GRU over 100 epochs. The confusion matrix in CNN-18-GRU confusion matrix figure describes class-wise test results of the CNN18-GRU. 

</p>

CNN-18-GRU training and validation accuracy over epochs            |  CNN-18-GRU confusion matrix
:-----------------------------------------------------------------:|:-----------------------------:
![iemocap_cnn18gru_acc](images/iemocap_cnn18gru_acc.png)  |  ![iemocap_cnn18gru_confusion_matrix_1](images/iemocap_cnn18gru_confusion_matrix_1.png)


### On TESS dataset
<p align="center"> 
  
Amoung our model's four versions performance, the best architecture of our model is CNN-18-GRU as it achieves the highest accuracy and F1-score, 
where it reaches  99.2% accuracy and 99% F1-score on the TESS dataset which is better compared to the state of-the-art methods.
The CNN-18-GRU training and validation accuracy over epochs figure shows the evolution of training and validation accuracy of the CNN-18-GRU over 100 epochs. The confusion matrix in CNN-18-GRU confusion matrix figure describes class-wise test results of the CNN18-GRU.  

</p>

CNN-18-GRU training and validation accuracy over epochs            |  CNN-18-GRU confusion matrix
:-----------------------------------------------------------------:|:-----------------------------:
![cnn18gru_acc](images/cnn18gru_acc.png)  |  ![cnn18gru_confusion_matrix](images/cnn18gru_confusion_matrix.png)

### On RAVDESS dataset
<p align="center"> 
  
Our CNN-18-GRU model also demonstrates excellent performance on the RAVDESS dataset, achieving an accuracy of 86.6% and an F1-score of 86.7%. The confusion matrix below shows the model's performance across the 8 emotion categories in RAVDESS, revealing strong discrimination capabilities particularly for angry, fearful, and surprised emotional states.

</p>

<div align="center">
  
![ravdess_cfm](images/ravdess_cfm.png)
  
*CNN-18-GRU confusion matrix on RAVDESS dataset*
  
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<p align="center">
  
_For more detailed experiments and results you can read the paper._
  
</p>


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

All source code is made available under a BSD 3-clause license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See [`LICENSE.md`](LICENSE.md) for the full license text.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Alaa Nfissi - [@LinkedIn](https://www.linkedin.com/in/alaa-nfissi/) - alaa.nfissi@mail.concordia.ca

Github Link: [https://github.com/alaaNfissi/CNN-n-GRU-for-speech-emotion-recognition](https://github.com/alaaNfissi/CNN-n-GRU-for-speech-emotion-recognition)

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[ser-approaches]: images/ser_aproaches.png
[model-architecture]: images/model_architecture.png


[anaconda.com]: https://anaconda.org/conda-forge/mlconjug/badges/version.svg
[anaconda-url]: https://anaconda.org/conda-forge/mlconjug

[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
