what is t ?
Why linear after conv 1x1 in bottleneck ?
How to merge residual if the shape is different ?
How does the input stay the same with stride = 2 at the start ?
What loss function is used ? Does this use paired data ? Is this supervised ?

Difference code and paper:
- Different init learning rate
- Have to activate Swa.


Output notes:
input/output 1 = 1500 epochs on 1 image, no post filter used

Models:
1 = just adam
2 = adam and swa
3 = updated model architeture by fixign typo and swa 
4 = trained on custom dataset

Validation on models:
1 = 0.922626
2 = 0.9115783
3 = 0.9385028
4 = 0.5172832

## todo
finish model <- done
build train script <- done
do filtering last
rewrite code to be more nice