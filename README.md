## Introduction and goal

In this blog an attempt has been made to reproduce the algorithm of one of the particiting teams of the OpenEDS Semantic Segmentation challenge of 2019. The challenge uses the OpenEDS dataset, which is created by facebook. The dataset contains grey images and eye segmentation masks (.npy files).  
The OpenEDS dataset exists out of 2.403 images with accompying mask in the validation set, 8.916 images with accompying mask in the training set, and 1.440 test images with accompying mask. In this paper the reproducibility of the model described in the paper Eye Semantic Segmentation with A Lightweight Model is researched[[5]](#5).

We tried to reconstruct their model using only published paper and through asking questions to the authors if we were ever stuck on an mather. Our goal was to do three things;
1. Recreate their model using only the paper
2. Use their Model uploaded to github and see if we get the same results
3. Create a new dataset and train our own model on it

First we will briefly talk about point 2 and 3 and then go a little more in depth for point 1.

## Using their model
During the attempt to reproduce the project we used their github project. However, there were multiple issues we had with their uploaded code. Firstly, it did not include any documentation about which version of the packages were used. Secondly, when enabling the Stochastic Weighted Averaging option, as used in the paper, the training script crashed at every fifth epoch when testing the current model with the validation set. The most probably cause for this is that the authors cloned an official torch package and modified it to match their model, this is nowhere mentioned and the additional code is not provided. It gave the message that the dimensions of the input were not right. Additionally, the training script did not save the checkpoints. The script should have saved evert 25th checkpoint of the model. However, only the model at the 0th, 25th and 50th epoch were saved. Lastly, after multiple attempts training the full 200 epochs, it did not save the model at the end. Due to time and budget limits we eventually had to abandon the idea of training a working model from scratch with their code.

## Creating a new dataset
Besides the OpenEDS dataset a different dataset was used for training. The SBVI dataset contains a mask for the sclera, pupil and iris of 3000x1700 colour images of eyes[[1]](#1),[[2]](#2),[[3]](#3),[[4]](#4). This image was resized to the dimensions 400x640 and converted to grey scale. Futhermore, the images that contained the masks of the three different classes were combined, resized and converted to the correct format for the masks (.npy). This resulted in 122 images with an accompanying mask. These images and maks were flipped to generate a total of 244 images that we could use to train our model.

## Reproducing the model
When recreating the model we tried to do it in the following steps, first; We create the model, secondly we write a training script, third we write the code for the post processing and fourth we write the script that validates the model. 

### Encoder
The structure of the model is clearly defined in the paper using figures and tables, we first looked at the encoder part of the model, which looks like this:

![image](https://user-images.githubusercontent.com/9881502/122454321-8ef96b80-cfab-11eb-98a4-07edd083d1be.png)[[5]](#5)

Immediatly, a few questions arose; what does the linear after the conv 1x1 mean in the block, what are the variables t,c,n and s in the table, why does the input size not decrease when the first layer has stride 2. We first tried creating a single bottleneck block and tried to run the code with just a random torch matrix with shape (320,200). It immediatly crashed because I added an linear activation function after my conv 1x1, after doing some research having a conv 1x1 linear is the same as just doing a conv 1x1 without using a activation function. So when I removed the linear activation functiones after my conv 1x1 the bottleneck block worked. But this is without using the variables mentioned in the table, according to the table the encoder begins with a conv2d layer has 9 bottleneck blocks and ends with a conv2d layer. The n in the table corresponds to this amount of 9, we start with n=2 bottlenecks with values; t = 1, c = 16 and s = 1. After doing some research we found out what the variables ment; s is used to indicate stride, c is used to indicate the dimensions of the input and t was the expansion factor which upscales the input before downscaling it again. Using this info we chained the bottlenecks together and tried to recreate the tables input sizes. But the first conv2d has a stride of 2 while the input size stays the same, so there was an mistake, either the stride on the first conv2d should be an 1 or the input sizes are incorrect. We first assumed the first case but after contacting the authors they mentioned it was an typo in the paper and the input sizes are incorrect and it should infact shrink the input size. We trained models based on both of these changes.

### Decoder
The figure used to denote the decoder looks like this:

![image](https://user-images.githubusercontent.com/9881502/122454343-94ef4c80-cfab-11eb-8070-8ecca43dedb5.png)[[5]](#5)

On first glance, the decoder looks a lot more complicated because of all the merging involved. But the operations are very clearly defined and we didn't run into any issues while developing it. The SE Block was a known methode which we could very easily google. After developing the decoder and testing the input and output sizes using random matrices with the shapes specified in the paper, we were ready to chain the encoder and decoder together and build the training script.

## Building the training script
Building the training script there were 2 major roadblocks, firstly the structure of the data and secondly the loss function.
Building a loop that reads data from a dataloader and throws it into a loss function was done relativly fast, but the dataloader should also return the groundtruth so that we can use it for the loss function. And we also have to implement this loss function, which in the paper is called a: generalized dice loss. 

### Generalized dice loss
A generalized dice loss is a loss function that uses the mean intersection-over-union(mIoU) but also includes weights, which makes it perfect for cases where there is class imbalance. Like, the segmentation of the eye, where a big part is the background and a smaller part is the pupil. For this loss function we compare the pixels between the groundtruth and the output of our decoder, which has a dimension of 4, one dimension for each class. {0, 1, 2, 3} corresponding to the background, sclera, iris and pupil. The groundtruth provided by openEDS were 640x400 matrices, where each pixel is labeled according to its class.

### Dataloader with groundtruth
To use this loss function, we should use a dataloader that not only returns the images but also returns the labels belonging to them. To do this we needed to create a custom dataset where we can implement the `getitem` function ourselves, In the end the `getitem` function looked like this:
```python
def __getitem__(self, index):
    img_path = self.list_png[index]
    npy_path = img_path.replace('images', 'labels').replace('png', 'npy')

    img = Image.open(img_path)
    npy = np.load(npy_path, allow_pickle=False)

    sample = {'image': img, 'mask': npy}
    if self.transform:
        sample["image"] = self.transform(img)

    return sample
```
adding this and the loss function made it possible to train the model and get good results. But the paper does a post processing step that improves the accuracy of the model.

## Post processing; Heuristic filtering
The algorithm to process the output of the decoder is provided:

![image](https://user-images.githubusercontent.com/9881502/122459323-172e3f80-cfb1-11eb-8f03-aa1c1e11b34d.png)[[5]](#5)

The algorithm uses the predicted mask and runs through all classes and fills in the gaps, and updates the location of the gaps in the predicted mask so it gets ignored in the upper classes. So if the predicted mask labels a piece of the background as iris, the gap will be filled while looking at the background and the predicted mask will be updated aswell. The part causing us trouble was the sentence "Fill all black holes", with no clear way of how to excatly do this. We ended up using cv2 floodfill and eroding the edges of the images to fill in the holes. It works pretty well but probably not as good as what they used. Our code to fill in the holes looks like this, the input is the BW of the corresponding class:
```python
def fill_hole(input_mask):
    h, w = input_mask.shape
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    canvas[1:h + 1, 1:w + 1] = input_mask.copy()

    mask = np.zeros((h + 4, w + 4), np.uint8)

    cv2.floodFill(canvas, mask, (0, 0), 1)
    cv2.floodFill(canvas, mask, (0, 639), 1)
    canvas = canvas[1:h + 1, 1:w + 1].astype(np.bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.dilate((~canvas | input_mask.astype(np.uint8)) , kernel)
    eroded=cv2.erode(dilated,kernel)
    return eroded 
```
And showing results:

![image](https://raw.githubusercontent.com/AsWali/eye-segmentation/main/tests/output3.png?token=ACLMPHXNIR6XP6CFOELCTWDA2TPSU)

![image](https://raw.githubusercontent.com/AsWali/eye-segmentation/main/tests/filtered_output3.png?token=ACLMPHT7CZVZG54P3U5ZMI3A2TPTY)


## Validation and results
To validate our model we created a script that looks at all files in the validation set and compares our filtered predicted mask with the groundtruth. It again uses mIoU to score the model, the results reached by the published model looks like this:

![image](https://user-images.githubusercontent.com/9881502/122460302-3d081400-cfb2-11eb-9d07-420c116461ed.png)

We will only look at the mIoU to compare accuracies.

Validation on models: 

In total we trained 6 models:
1. openEDS, only uses adam optimizer, 200 epochs
2. openEDS, uses adam and swa, 200 epochs
3. openEDS, used the correct input size in the architecture, 200 epochs
4. SBVI, 200 epochs
5. SBVI, 500 epochs
6. SBVI, 1500 epochs

For the SBVI dataset we didn't have a validation set since our data was limited, so we scored the accuracy on the training set. However, it should be carefully noticed that this causes overfitting on the dataset which results in higher mIoU accuracies.
The accuracies reached for the models were as follows:
1. 0.922626 
2. 0.9115783 
3. 0.9385028 
4. 0.5172832 
5. 0.93217456 
6. 0.9556759

For the openEDS dataset we got pretty close, their highest score is 0,9485 while ours is 0,9385. This change is high likely because of their post process filter, it's probably more sufficted than ours and works better. The accuracies reached on our own dataset increased a lot based on the epochs, it did learn the images and we even used an image of an eye from wikipedia and this was the result:

![image](https://raw.githubusercontent.com/AsWali/eye-segmentation/main/tests/c_input2.png?token=ACLMPHXEVQRDIF4DJON3O33A2TPRQ)

![image](https://raw.githubusercontent.com/AsWali/eye-segmentation/main/tests/c_output2.png?token=ACLMPHR6BYDXK3UE5ZFLEZ3A2TPNO)

![image](https://raw.githubusercontent.com/AsWali/eye-segmentation/main/tests/c_output3.png?token=ACLMPHQVZQ6TZTBT3ACUUJDA2TPQO)

The last picture is the post process filter, which goes completly wrong in this example. Due time constraint we didn't have the time to debug this issue and fix it. But this further shows that our implementation of the post process filter has a flaw in it, which also affects the accuracy reached on the openEDS validation set.


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
