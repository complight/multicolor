[![](https://www.replicabilitystamp.org/logo/Reproducibility-small.png)](http://www.replicabilitystamp.org#https-github-com-complight-multicolor)

# Multi-color Holograms Improve Brightness in Holographic Displays 
[Koray Kavaklı](https://www.linkedin.com/in/koray-kavakli-75949241/),
[Liang Shi](https://people.csail.mit.edu/liangs/),
[Hakan Ürey](https://mems.ku.edu.tr/),
[Wojciech Matusik](https://cdfg.csail.mit.edu/wojciech),
and [Kaan Akşit](https://kaanaksit.com)

<img src='./media/teaser.png' width=800>


[\[Website\]](https://complightlab.com/publications/multi_color) 


## Description
Holographic displays generate Three-Dimensional (3D) images by displaying single-color holograms time-sequentially, each lit by a single-color light source.
However, representing each color one by one limits brightness in holographic displays.

This paper introduces a new driving scheme for realizing brighter images in holographic displays.
Unlike the conventional driving scheme, our method utilizes three light sources to illuminate each displayed hologram simultaneously at various intensity levels.
In this way, our method reconstructs a multiplanar three-dimensional target scene using consecutive multi-color holograms and persistence of vision.
We co-optimize multi-color holograms and required intensity levels from each light source using a gradient descent-based optimizer with a combination of application-specific loss terms.
We experimentally demonstrate that our method can increase the intensity levels in holographic displays up to three times, reaching a broader range and unlocking new potentials for perceptual realism in holographic displays.


### Citation
__If you find this repository useful for your research, please consider citing our work using the `BibTeX entry` in the [project website](https://complightlab.com/publications/multi_color).__


## Getting started
This repository contains a code base for calculating holograms that can be used to generate higher dynamic range holograms.
These holograms are calculated such that they can be illuminated with multiple wavelengths.


### (0) Requirements
You can clone our codebase by typing:

```shell
git clone git@github.com:complight/multicolor.git
```

Before using this code in this repository, please make sure to have the right dependencies installed.
In order to install the main dependency used in this project, please make sure to use the below syntax in a Unix/Linux shell:

```shell
cd multicolor
pip3 install -r requirements.txt
```

Note that we often update `odak`, if this `requirements.txt` fails, please use the below syntax to install odak:

```shell
pip3 install odak
```


### (1) Runtime
Once you have the main dependency installed, you can run the code base using the default settings by providing the below syntax:

```shell
python3 main.py
```


### (2) Reconfiguring the code for your needs
Please consult the settings file found in `settings/jasper.txt`, where you will find a list of self descriptive variables that you can modify according to your needs.
This way, you can create a new settings file or modify the existing one.

If you are willing to use the code with another settings file, please use the following syntax:

```shell
python3 main.py --settings settings/jasper.txt
```

If you are looking into finding more sample images, consider visiting our [images repository](https://github.com/complight/images).


## Support
For more support regarding the code base, please use the issues section of this repository to raise issues and questions.
