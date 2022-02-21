<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">Skelevision</h3>

  <p align="center">
    A deep neural network for high throughput measurement of functional traits on museum skeletal specimens.
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#getting-data-and-model">Getting Data and Model</a></li>
        <li><a href="#measuring-bones">Measuring Bones</a></li>
        <li><a href="#training">Training</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


![CV Workflow](media/cv_workflow.jpg)



<!-- ABOUT THE PROJECT -->
## About The Project

We adapt recent deep nerual network approaches in computer vision to enable high throughput measurement of functional traits on museum skeletal specimens. This repository will provide code and link to bird skeleton images. We hope that this will accelerate and inspire large scale study of functional traits on museum skeletal specimens. 

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This section will guide you through installing the required dependencies and running the Skelevision model. 

### Prerequisites

We recommend creating a conda environment and installing the required prerequisites there. 

To install miniconda:

- Visit [miniconda](https://docs.conda.io/en/latest/miniconda.html) website
- Download the corresponding .sh file for your system
- Linux:
    - ```chmod +x {Miniconda3-latest-Linux-x86_64.sh}```
    - ```./ {Miniconda3-latest-Linux-x86_64.sh}```
    - ```export PATH="/home/{username}/miniconda/bin:$PATH"```
    - ```source ~/.zshrc```

Make sure to replace the file names in {} with the right ones for your installation. Verify the installation of conda by typing "conda -V" in the command prompt, which should show the conda version installed. 

Create a new conda environment:

- ```conda create --name skelevision-env python=3.8```
- ```conda activate skelevision-env```

We require the installation of the following dependicies from their respective websites:

- PyTorch (https://pytorch.org/get-started/locally/)
- Detectron2 (https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

We recommend installing GPU versions of PyTorch / Detectron2.

### Installation

After installing PyTorch and Detectron2, we need to install a few more dependencies.

```pip install -r requirements.txt```


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

This section will guide you through downloading the dataset, the pretrained model, and measuring bones. 

### Getting Data and Model

The data will be available on https://skelevision.net/. Please download the data and extract the data in the /data/ such that the structure from the directory looks like this. 

The pretrained model will be available here. Please place the model_final.pth in the following structure shown below. 

- data
    - Training_Images
        - UMMZ_22510
        - UMMZ_
    - processed_images (created after running process_data.py)
        - 22510.jpg
        - ******.jpg
- models
    - oct_2021
        - model_final.pth
- annotations
    - oct_2021_annotations.json
- README.md
- predict.py
- ...

Finally, run the script to process data:

```python process_data.py```

### Measuring Bones

After setting up the data and model folders above, we can run some predictions. 

```python predict.py -m models/oct_2021 -d data/processed_images -g -1```

The parameters to predict.py are below:

- -m [model checkpoint folder] REQURIED, folder that contains model_final.pth
- -d [image directory] REQUIRED, we will predict on all images in this directory
- -o [output directory] DEFAULT, output/
- -g [gpu] DEFAULT -1 for CPU, or GPU id
- -c [pixel to mm conversion constant] DEFAULT 0.2309mm/pixel

Do not change -c when using data provided by skelevision.net. 

### Training

Coming soon.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

[skelevision.net](https://skelevision.net/)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

[skelevision.net](https://skelevision.net/)

<p align="right">(<a href="#top">back to top</a>)</p>



Create with [README Page Template](https://github.com/othneildrew/Best-README-Template/blob/master/README.md)




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/shadowninjazx/skelevision/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/shadowninjazx/skelevision/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/shadowninjazx/skelevision/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/shadowninjazx/skelevision/issues
[product-screenshot]: images/screenshot.png
