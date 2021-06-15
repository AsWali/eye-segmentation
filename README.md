## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/AsWali/eye-segmentation/edit/main/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

In this blog an attempt has been made to reproduce the algorithm of one of the particiting teams of the OpenEDS Semantic Segmentation challenge of 2019. The challenge uses the OpenEDS dataset, which is created by facebook. The dataset contains grey images and eye segmentation masks (.npy files).
In this paper the reproducibility of the model described in the paper Eye Semantic Segmentation with A Lightweight Model[[5]](#5).


The OpenEDS dataset exists out of 2.403 images with accompying mask in the validation set, 8.916 images with accompying mask in the training set, and 1.440 images with accompying mask in the 



During the attempt to reproduce the project we used their github project. However, there were multiple problems with their github. Firstly, it did not include any documentation about which version of the packages were used. Secondly, when enabling the Stochastic Weighted Averaging option, as used in the paper, the training script crashehd at every fifth epoch when testing the current model with the validation set. It gave the message that the dimensions of the input were not right. Additioanally, the training script did not save the checkpoints. The script should have saved evert 25th checkpoint of the model. However, only the model at the 0th, 25th and 50th epoch were saved. Lastly, after mutiple attempt training the full 200 epochs, it did not save the model at the end. Due to time and budget limits we eventually had to abandon the idea of training a working model from scratch with their code.

Besides the OpenEDS dataset a different dataset was used for training. The SBVI dataset contains a mask for the sclera, pupil and iris of 3000x1700 colour images of eyes[[1]](#1),[[2]](#2),[[3]](#3),[[4]](#4). This image was resized to the dimensions 400x640 and converted to grey scale. Futhermore, the images that contained the masks of the three different classes were combined, resized and converted to the correct format for the masks (.npy). This resulted in 122 images with an accompanying mask. These images and maks were flipped to generate a total of 244 images that were used for training, validation and testing.
### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. I```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/AsWali/eye-segmentation/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
```
"...the **go to** statement should be abolished..." [[1]](#1).

## References
<a id="1">[1]</a> 
Vitek, Matej and Rot, Peter and Štruc, Vitomir and Peer, Peter  (2020). 
doi = "10.1007/s00521-020-04782-1"
A Comprehensive Investigation into Sclera Biometrics: A Novel Dataset and Performance Study, 1-15.

<a id="2">[2]</a> 
Rot, Peter and Vitek, Matej, and Grm, Klemen and  Emeršič, Žiga, and Peer, Peter and  Štruc, Vitomir(2019),
isbn = "978-3-030-27731-4",
doi = "10.1007/978-3-030-27731-4_13",
Deep Sclera Segmentation and Recognition, 395-432.


<a id="3">[3]</a> 
Rot, Peter and Vitek, Matej, and Grm, Klemen and  Emeršič, Žiga, and Peer, Peter and  Štruc, Vitomir(2018),
doi = "10.1109/IWOBI.2018.8464133",
Deep Multi-class Eye Segmentation for Ocular Biometrics, 1-8.

<a id="4">[4]</a> 
Vitek, Matej and Das, Abhijit and Pourcenoux, Yann and Missler, Alexandre and Paumier, Calvin and Das, Sumanta and De Ghosh, Ishita and Lucio, Diego R. and Zanlorensi Jr., Luiz A. and Menotti, David and Boutros, Fadi and Damer, Naser and Grebe, Jonas Henry and Kuijper, Arjan and Hu, Junxing and He, Yong and Wang, Caiyong and Liu, Hongda and Wang, Yunlong and Sun, Zhenan and Osorio-Roig, Daile and Rathgeb, Christian and Busch, Christoph and Tapia Farias, Juan and Valenzuela, Andres and Zampoukis, Georgios and Tsochatzidis, Lazaros and Pratikakis, Ioannis and Nathan, Sabari and Suganya, R and Mehta, Vineet and Dhall, Abhinav and Raja, Kiran and Gupta, Gourav and Khiarak, Jalil Nourmohammadi and Akbari-Shahper, Mohsen and Jaryani, Farhang and Asgari-Chenaghlu, Meysam and Vyas, Ritesh and Dakshit, Sristi and Dakshit, Sagnik and Peer, Peter and Pal, Umapada and Štruc, Vitomir (2020),
{SSBC} 2020: Sclera Segmentation Benchmarking Competition in the Mobile Environment.

<a id="5">[5]</a> 
Huynh, V.T. and Kim, S. and Lee, G. and Yang, H. (2019). 
Eye Semantic Segmentation with A Lightweight Model, 3694-3697.

<a id="6">[6]</a> 
Huynh, V.T. and Kim, S. and Lee, G. and Yang, H. (2020)
Semantic Segmentation of the Eye With a Lightweight Deep Network and Shape Correction, 131967-131974.
