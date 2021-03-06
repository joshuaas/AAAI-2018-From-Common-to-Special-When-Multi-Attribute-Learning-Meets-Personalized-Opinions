### From Common to Special: When Multi-Attribute Learning Meets Personalized Opinions
![illu](illu1.png)

---
#### Discription
This is a matlab implementation of our AAAI 2018 paper: 

*Zhiyong Yang, Qianqian Xu, Xiaochun Cao, Qingming Huang:From Common to Special: When Multi-Attribute Learning Meets Personalized Opinions. AAAI 2018: 515-522*

If you use our code, please cite our with the following bibtex code :

    @inproceedings{Yang2018FromCT,

              title={From Common to Special: When Multi-Attribute Learning Meets Personalized Opinions},
 
              author={Zhiyong Yang and Qianqian Xu and Xiaochun Cao and Qingming Huang},
 
              booktitle={AAAI},
 
              year={2018}
 
     }

---

#### Acknowledgements
We would like to thank Dr. Jiayu Zhou for providing the [MALSAR package](https://github.com/jiayuzhou/MALSAR) 

---
#### Objective Function
<img src="gif.gif" />

---
#### Paramters

|Parameters| Discription|
| ------ | ------ |
|  X |  A cell with scale num_t \* num_u, where num_t is the number of attributes, and num_u represents the number of annotaters for each attribute.Each element of X (X{i,j}) is an input feature matrix with scale num_sample * num_feature. |
|  Y |   A cell with scale num_t \* num_u, where num_t is the number of attributes, and num_u    represents the number of annotaters for each attribute.Each Y (Y{i,j}) is an input label vector with scale num_sample * 1.  |
|lambda1  | see the objective function |
|lambda2  | see the objective function |
|lambda3  |  see the objective function |
|opts  |  The same as MALSAR  |
