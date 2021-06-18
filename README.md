## Introduction and goal

In this blog, an attempt has been made to reproduce the algorithm of one of the participating teams of the OpenEDS Semantic Segmentation challenge of 2019. The challenge uses the OpenEDS dataset, which is created by Facebook. The dataset contains greyscale images and eye segmentation masks (.npy files).  These segmentaiton masks contain the sclera, pupil, iris and background.
The OpenEDS dataset exists out of 2.403 images with the accompanying mask in the validation set, 8.916 images with the accompanying mask in the training set, and 1.440 test images with the accompanying mask. In this paper, the reproducibility of the model described in the paper "Eye Semantic Segmentation with A Lightweight Model" is researched[[5]](#5).

We tried to reconstruct their model using only their published paper and through asking questions to the authors if we were ever stuck on a mather. Our goal was to do three things:
1. Recreate their model using only the paper;
2. Use their model for which the training and test script was uploaded to GitHub to see if we get the same results;
3. Create a new dataset and train our own model on it.

First, we will briefly discuss points 2 and 3 and then go a bit further in detail on point 1.

## Using their model
During the attempt to reproduce the project we used their Github project. However, there were multiple issues we had with their uploaded code. Firstly, it did not include any documentation about which versions of the packages were used. Secondly, when enabling the Stochastic Weighted Averaging option, as used in the paper, the training script crashed at every fifth epoch when testing the current model with the validation set. The most probable cause for this is that the authors cloned an official torch package and modified it to match their model, this is nowhere mentioned and the additional code is not provided. It gave the message that the dimensions of the input were not right. Additionally, the training script did not save the checkpoints. The script should have saved a checkpoint of the model every 25th epoch during training. However, only the model was saved at the 0th, 25th, and 50th epoch. Lastly, after multiple attempts to train the full 200 epochs, it did not save the model at the end. Due to time and budget limits we eventually had to abandon the idea of training a working model from scratch with their code.

## Creating a new dataset
Besides the OpenEDS dataset, a different dataset was used for training. The SBVI dataset contains a mask for the sclera, pupil and iris of 3000x1700 colour images of eyes[[1]](#1),[[2]](#2),[[3]](#3),[[4]](#4). This image was resized to the dimensions 640x400 and converted to greyscale. Furthermore, the images that contained the masks of the three different classes were combined, resized, and converted to the correct format for the masks (.npy). These masks have a dimension of 4, one dimension for each class. {0, 1, 2, 3} corresponding to the background, sclera, iris, and pupil. The ground truths were 640x400 matrices, matching the size of the OpenEDS dataset, where each pixel is labeled according to its class. This resulted in 122 images with an accompanying mask. These images and maks were flipped to generate a total of 244 images that we could use to train our model.

## Reproducing the model
When recreating the model we tried to do it in the following steps, sequence of steps; We started with creating the model. Secondly, we wrote a training script, followed by the pos-processing. Lastly, we wrote the script that validates the model.

### Encoder
The structure of the model is clearly defined in the paper using figures and tables, we first looked at the encoder part of the model, which looks like this[[5]](#5):

![image](https://user-images.githubusercontent.com/9881502/122454321-8ef96b80-cfab-11eb-98a4-07edd083d1be.png)

Immediately, a few questions arose; what does the linear after the conv 1x1 mean in the block, what are the variables t,c,n, and s in the table, why does the input size not decrease when the first layer has stride 2. We first tried creating a single bottleneck block and tried to run the code with just a random torch matrix with shape (320,200). The code immediately crashed because we added a linear activation function after the conv 1x1. After doing some research we realized that a conv 1x1 linear is the same as just doing a conv 1x1 without using an activation function. Then, when the linear activation functions had been removed after the conv 1x1, the bottleneck block worked. However, this is done without using the variables mentioned in the table. According to the table, the encoder begins with a conv2d layer has 9 bottleneck blocks and ends with a conv2d layer. The n in the table corresponds to this amount of 9, we start with n=2 bottlenecks with values; t = 1, c = 16 and s = 1. After doing some research we found out what the variables meant; s is used to indicate stride, c is used to indicate the dimensions of the input and t was the expansion factor that upscales the input before downscaling it again. Using this info, we chained the bottlenecks together and tried to recreate the table's input sizes. However, the first conv2d has a stride of 2 while the input size stays the same, which means there was a mistake. The mistake was that either the stride on the first conv2d should be a 1 or the input sizes are incorrect. We first assumed the first case but after contacting the authors they mentioned it was a typo in the paper and the input sizes are incorrect and it should, in fact, shrink the input size. We trained models based on both of these changes.

### Decoder
The figure [[5]](#5) used to denote the decoder looks like this:

![image](https://user-images.githubusercontent.com/9881502/122454343-94ef4c80-cfab-11eb-8070-8ecca43dedb5.png)

At first glance, the decoder looks a lot more complicated because of all the merging involved. However, the operations are very clearly defined and we did not run into any issues while developing it. The SE Block was a known method that was easy to look up on Google. After developing the decoder and testing the input and output sizes using random matrices with the shapes specified in the paper, we were ready to chain the encoder and decoder together and build the training script.

## Building the training script
Building the training script there were 2 major roadblocks. The first roadblock was the structure of the data. The second roadblock was the loss function.
Building a loop that reads data from a dataloader and throws it into a loss function was done relatively fast, but the dataloader needed also to return the ground truth. The reason for this is that it can be used for the loss function. Furthermore, we also had to implement this loss function, which in the paper is called a: "generalized dice loss"[[5]](#5). 

### Generalized dice loss
A generalized dice loss is a loss function that uses the mean intersection-over-union(mIoU) but also includes weights, which makes it perfect for cases where there is class imbalance. Like, the segmentation of the eye, where a big part is the background and a smaller part is the pupil. For this loss function, we compared the pixels between the ground truth and the output of our decoder, which has a dimension of 4, one dimension for each class. {0, 1, 2, 3} corresponding to the background, sclera, iris and pupil. The ground truths provided by OpenEDS were 640x400 matrices, where each pixel is labeled according to its class.

### Dataloader with ground truth
To use this loss function, we should use a dataloader that not only returns the images but also returns the labels belonging to them. To do this we needed to create a custom dataset where we can implement the `getitem` function ourselves. The final version of the `getitem` function looked like this:
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
After adding the 'getitem' function and the loss function it was possible to train the model and get good results. However,  the paper does a post-processing step that improves the accuracy of the model[[5]](#5).

## Post processing; Heuristic filtering
The algorithm to process the output of the decoder is provided:

![image](https://user-images.githubusercontent.com/9881502/122459323-172e3f80-cfb1-11eb-8f03-aa1c1e11b34d.png)[[5]](#5)

The algorithm uses the predicted mask and goes through all classes, filling in the holes, and updating the location of the gaps in the predicted mask, which means it gets ignored in the upper classes. So, for example, lets say a part of the background gets classified as 'iris', the algorithm will see that there is a gap in the segmentation of the background and overwrite this to background.
The part that caused us trouble was the sentence "Fill all black holes", with no clear way described on how to exactly perform this. We ended up using cv2 floodfill and eroding the edges of the images to fill in the holes[[5]](#5). It works pretty well but probably not as good as what they used. Our code to fill in the holes looks like this, the input is the BW of the corresponding class:
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
The results are:

![image](https://raw.githubusercontent.com/AsWali/eye-segmentation/main/tests/output3.png?token=ACLMPHXNIR6XP6CFOELCTWDA2TPSU)

![image](https://raw.githubusercontent.com/AsWali/eye-segmentation/main/tests/filtered_output3.png?token=ACLMPHT7CZVZG54P3U5ZMI3A2TPTY)


## Validation and results
To validate our model we created a script that looks at all files in the validation set and compares our filtered predicted mask with the ground truth. It again uses mIoU to score the model, the results reached by the published model look like this[[5]](#5):
![image](https://user-images.githubusercontent.com/9881502/122460302-3d081400-cfb2-11eb-9d07-420c116461ed.png)

While the equation used to do the evaluation looks like this:

![image](https://user-images.githubusercontent.com/9881502/122557594-48ece800-d03d-11eb-94c9-2549feb9ade2.png)

This was hard to understand, there was an magic number 50 in the equation. Comparing our trainable params to the one in the paper we also got different results(three times as large). Due to time constraint we couldn't really find the root problem of this, we do have a strong feeling it has something to do with the depthwise separable convolutions. Our code works and we implemented it the way the paper specified it, but there must be something we didn't see. But looking at their code they don't use the trainable parms for the evaluation, they only calculate the mIoU. Hence, we decided to only look at the mIoU to evaluate the accuracy.

Validation on models: 

In total we trained 6 models:
1. OpenEDS, only uses adam optimizer, 200 epochs;
2. OpenEDS, uses adam and swa, 200 epochs;
3. OpenEDS, used the correct input size in the architecture, 200 epochs;
4. SBVI, 200 epochs;
5. SBVI, 500 epochs;
6. SBVI, 1500 epochs.

For the SBVI dataset we did not have a validation set since our data was limited. We decided to score the accuracy on the training set. However, it should be carefully noticed that this often causes overfitting on the dataset which results in higher mIoU accuracies compared to when it would be verified by using unseen images with ground truths from the same distribution.

The mIoU accuracies reached for the models were as follows:
1. 0.922626 
2. 0.9115783 
3. 0.9385028 
4. 0.5172832 
5. 0.93217456 
6. 0.9556759

Looking at the results for the OpenEDS dataset, it can be concluded we got rather close to the mIoU accuracy of the paper. Their highest score is 0,9485 while ours is 0,9385. This change is highly likely because of their post-process filter, which is probably more sophisticated than ours and has higher performance. The accuracies reached on our own dataset increased a lot based on the epochs, it did learn the images and we even used an image of an eye from Wikipedia and this was the result:

![image](https://raw.githubusercontent.com/AsWali/eye-segmentation/main/tests/c_input2.png?token=ACLMPHXEVQRDIF4DJON3O33A2TPRQ)

![image](https://raw.githubusercontent.com/AsWali/eye-segmentation/main/tests/c_output2.png?token=ACLMPHR6BYDXK3UE5ZFLEZ3A2TPNO)

![image](https://raw.githubusercontent.com/AsWali/eye-segmentation/main/tests/c_output3.png?token=ACLMPHQVZQ6TZTBT3ACUUJDA2TPQO)

The last picture shows the post-process filter, which goes completely wrong in this example. Due to time constraints, we did not have the time to debug this issue and fix it. However, t this further shows that our implementation of the post-process filter has a flaw in it, which also affects the accuracy reached on the OpenEDS validation set.

## Discussion
In this blog, different attempts were discussed to check the reproducibility of the paper. Due to the paper being incomplete and missing elaboration on aspects as pre-processing. Furthermore, because the source code was broken and due to budget and time constraints, we could not check if the exact same results could be achieved when using their code to train from scratch. Additionally, in this blog, the results are shared of our own implementation of their model. It may not be an exact match of the model of the authors, but it is still quite similar. It is shown that we came relatively close to the mIoU accuracies described in the paper[[5]](#5).

Lastly, this paper showed that we created our own custom dataset using the SBPI dataset. Due to the limited amount of masks available, this only resulted in 122 images with ground truths. By the use of the data augmentation method of flipping, we doubled the dataset to 244 images. However, due to the small size of the dataset, we did not have a validation set and used the training set as a validation set. However, it should be noticed that this often results in overfitting on the data, resulting in exceptionally high accuracies compared to when it would be verified by using unseen images with ground truths from the same distribution.

## Conslusion
From the validation and test results, it can be concluded that we succeeded in reproducing the model described in the paper. Furthermore, due to the mistakes in the code of the authors and in the paper, it is relatively hard to validate the results described in their paper. This significantly affected the ability to reproduce the paper. However, it can be concluded that we successfully validated their approach for eye segmentation by recreating their model from scratch as similar as possible. Furthermore, we researched the possibility of creating our own dataset using an available dataset. This was successfully done by combining the ground truths of the SVPI dataset.

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
