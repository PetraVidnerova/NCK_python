# NCK_python 

Repository of General-purpose Library of ML/AI Methods.

The important part of successful neural network solution is a choice of an appropriate architecture. We provide tools automating this process, including semi-manual tools and fully automatic search. The semi-manual solution provides tools for automatic evaluation of a set of user defined architectures. 
The fully automatic search is based on evolutionary optimisation that finds a suitable network for a given problem. 

GANs ...


### Authors 
D. Coufal, F. Hakl, P. Vidnerová. 
The Czech Academy of Sciences, Institute of Computer Science

### Keywords 
deep neural networks, generative adverisal networks, conditional generation, generative algorithms, neural architecture search, model selection, evolutionary algorithms, multiobjective optimisation

### Contents
```
|
|__ NAS  (Neural Architecture Search tools) 
|     |
|     |__ semi_manual   
|     |            |
|     |            |__ data         (data preprocessing for pytorch) 
|     |            |
|     |            |__ net_search   (scripts for automatic network evaluation)
|     |            |
|     |            |__ examples     (examples of config files)
|     |
|     |__ automatic 
|               |
|               |__ ananas    (scripts for automatic architecture search)
|               |
|               |__ examples  (examples of config files)
|                
|__ GANs 

```

See the individual subdirectories for details on the individual parts and corresponding user instructions.


### Acknowledgement
This work was partially supported by the TAČR grant TN01000024 and institutional support of the Institute of Computer Science RVO 67985807.
