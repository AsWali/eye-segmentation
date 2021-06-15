## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/AsWali/eye-segmentation/edit/main/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

In this blog an attempt has been made to reproduce the algorithm of one of the particiting teams of the OpenEDS Semantic Segmentation challenge of 2019. The challenge uses the OpenEDS dataset, which is created by facebook. The dataset contains grey images and eye segmentation masks (.npy files).



During the attempt to reproduce the project we used their github project. However, there were multiple problems with their github. Firstly, it did not include any documentation about which version of the packages were used. Secondly, when enabling the Stochastic Weighted Averaging option, as used in the paper, the training script crashehd at every fifth epoch when testing the current model with the validation set. It gave the message that the dimensions of the input were not right. Additioanally, the training script did not save the checkpoints. The script should have saved evert 25th checkpoint of the model. However, only the model at the 0th, 25th and 50th epoch were saved. Lastly, after mutiple attempt training the full 200 epochs, it did not save the model at the end. Due to time and budget limits we eventually had abandon the idea of training working modoel from scratch with their code.
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
