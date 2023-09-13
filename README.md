  # A Study of Architecture Optimization Techniques for Convolutional Neural Networks


This repository contains the code for the experiments conducted as part of the work 'A Study of Architecture Optimization Techniques for Convolutional Neural Networks' (https://doi.org/10.1007/978-3-031-37720-4_25).

All our experiments are reproducable. 

#### Our Pytorch environment configuration:
* python 3.10.9
* pytorch 1.13.1
* pytorch-lightning 1.9.1
* torchvision 0.14.1
* fvcore 0.1.5.post20221221

## Abstract

Deep Convolutional Neural Networks (CNNs) have become dominant in computer vision in recent years. However, due to their complexity, state-of-the-art CNN architectures are limited to high-performance computing environments. Edge devices such as smartphones or embedded computing platforms require a resource-aware approach. Therefore, it is often necessary to modify CNN models to make them compatible with the limited infrastructure. It is challenging to do this deliberately and effectively, given the many techniques proposed in the literature. We have empirically evaluated many of them on ResNet-101 and VGG-19 architectures. Our main contribution is the ablation study of how different approaches affect the final results in terms of the reduced number of model parameters, FLOPS, and unwanted accuracy drop. We have divided these methods into two groups: architecture compression and post-training compression. The first group concerns solutions, i.e., depth separable convolution, shuffle mechanism, or ghost module, which more or less interfere with the low- and high-level model structure. We have shown how to implement these methods for different models. We found that their impact depends on the architecture and the problem to be solved. The second group includes pruning and quantization, and we have studied them on different models and sparsity levels.

## Experiments results

### Optimization techniques for ResNet-101 trained on CIFAR-10


|           Approach               | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
| -------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|         Fire modules             |  -  |  ✓  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |
|       No FC classifier           |  -  |  -  |  -  |  -  |  -  |  -  |  ✓  |  ✓  |  ✓  |  -  |  ✓  |  ✓  |  ✓  |  ✓  |
|        Nx1 - 1xN Conv.           |  -  |  -  |  ✓  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |
|            DWconv                |  -  |  -  |  -  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |
|         Shuffle mech.            |  -  |  -  |  -  |  -  |  ✓  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  ✓  |  ✓  |
|       Inv. bottlenecks           |  -  |  -  |  -  |  -  |  -  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |
|      Optimization before GAP     |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  ✓  |  ✓  |  -  |  ✓  |  ✓  |  ✓  |  ✓  |
| Less channels on the early stage |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  ✓  |  -  |  ✓  |  ✓  |  ✓  |  ✓  |
|        Ghost bottleneck          |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |
|         SE connections           |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  ✓  |  -  |  ✓  |
|       **Parameters [M]**         |42.513 |59.061 |35.397 |21.240 | 7.536|2.255 |1.187 |0.873 |0.869 |0.450 |0.400 |1.424 |0.151 |1.174 |
|       **Parameters [%]**         |100.00 |138.92 | 83.26 | 49.96 | 17.73| 5.30 | 2.79 | 2.05 | 2.04 | 1.06 | 0.94 | 3.35 | 0.35 | 2.76 |
|         **FLOPS [M]**            |2520.22|3480.01|2141.58|1279.53|420.75|79.21 |62.39 |57.37 |53.00 |32.83 |29.76 |31.48 |12.80 |14.52 |
|         **FLOPS [%]**            |100.00 |138.08 | 84.98 | 50.77 |16.69 | 3.14 | 2.48 | 2.28 | 2.10 | 1.30 | 1.18 | 1.25 | 0.51 | 0.58 |
|         **Size [MB]**            |162.77 |225.41 |135.64 | 81.62 |29.36 | 8.96 | 4.87 | 3.65 | 3.64 | 1.96 | 1.76 | 7.70 | 0.82 | 4.75 |
|        **Accuracy [%]**          | 94.33 | 92.78 | 94.24 | 93.03 |92.87 |93.09 |93.07 |93.01 |93.00 |92.70 |92.43 |92.55 |90.82 | 90.74 |


### Optimization techniques for VGG-19 trained on CIFAR-100


|           Approach          | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| --------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|             GAP             |  -  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |
|        Nx1 - 1xN Conv.      |  -  |  -  |  ✓  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |
|            DWconv           |  -  |  -  |  -  |  ✓  |  -  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |
|       Fire modules          |  -  |  -  |  -  |  -  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |
|      No FC classifier       |  -  |  -  |  -  |  -  |  -  |  -  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |
|         Shuffle mech.       |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  ✓  |  -  |  ✓  |  ✓  |  ✓  |
|        Ghost modules        |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  ✓  |  ✓  |  ✓  |  -  |
|         SE connections      |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  ✓  |  ✓  |
|       **Parameters [M]**    |39.328 |20.087|13.150 |2.332 |1.848|0.520|0.520|0.198|0.443|0.388|1.588|1.398|
|       **Parameters [%]**    |100.00 |51.08 | 33.44 | 5.93 |4.70 | 1.32| 1.32|0.50 |1.13 |0.99 |4.04 |3.55 |
|         **FLOPS [M]**       |418.02 |398.79|256.08 | 46.8 |40.27|11.04|11.19|4.99 |9.62 |8.73 |10.23|6.49 |
|         **FLOPS [%]**       |100.00 |95.40 | 61.26 |11.20 |9.63 | 2.64| 2.68|1.19 |2.30 |2.09 |2.45 |1.55 |
|         **Size [MB]**       |150.10 |76.70 | 50.26 | 8.97 |7.21 | 2.14| 2.14|0.90 |1.85 |1.64 |6.23 |5.50 |
|        **Accuracy [%]**     | 71.91 |71.73 | 70.93 |66.26 |67.77|65.19|63.22|59.75|60.56|61.27|61.83|60.08|


### ResNet-101 trained on CIFAR-10 - pruning

<table>
    <thead>
        <tr>
            <th rowspan=2>Sparsity [%]</th>
            <th colspan=2>no. 1 - baseline</th>
            <th colspan=2>no. 11</th>
            <th colspan=2>no. 13</th>
        </tr>
        <tr>
            <th>Non-zeroed</th>
            <th>Acc [%]</th>
            <th>Non-zeroed</th>
            <th>Acc [%]</th>
            <th>Non-zeroed</th>
            <th>Acc [%]</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>0.0</td>
            <td>42 512 970</td>
            <td>94.33</td>
            <td>400 115</td>
            <td>92.43</td>
            <td>150 715</td>
            <td>90.82</td>
        </tr>
        <tr>
            <td>20.0</td>
            <td>33 930 189</td>
            <td>94.34</td>
            <td>304 678</td>
            <td>92.51</td>
            <td>105 273</td>
            <td>90.86</td>
        </tr>
        <tr>
            <td>36.0</td>
            <td>27 148 247</td>
            <td>94.36</td>
            <td>243 742</td>
            <td>92.41</td>
            <td>84 218</td>
            <td>90.51</td>
        </tr>
        <tr>
            <td>48.8</td>
            <td>21 722 694</td>
            <td>94.36</td>
            <td>194 994</td>
            <td>91.82</td>
            <td>67 374</td>
            <td>90.11</td>
        </tr>
        <tr>
            <td>59.0</td>
            <td>17 382 251</td>
            <td>94.39</td>
            <td>155 995</td>
            <td>91.79</td>
            <td>53 899</td>
            <td>89.85</td>
        </tr>
        <tr>
            <td>67.2</td>
            <td>13 909 897</td>
            <td>94.32</td>
            <td>124 796</td>
            <td>91.68</td>
            <td>43 119</td>
            <td>89.03</td>
        </tr>
        <tr>
            <td>73.8</td>
            <td>11 132 014</td>
            <td>94.36</td>
            <td>99 837</td>
            <td>91.21</td>
            <td>34 495</td>
            <td>87.85</td>
        </tr>
        <tr>
            <td>79.0</td>
            <td>8 909 707</td>
            <td>94.11</td>
            <td>79 870</td>
            <td>90.59</td>
            <td>27 596</td>
            <td>87.03</td>
        </tr>
        <tr>
            <td>83.2</td>
            <td>7 131 862</td>
            <td>94.16</td>
            <td>63 896</td>
            <td>90.19</td>
            <td>22 077</td>
            <td>84.88</td>
        </tr>
        <tr>
            <td>86.5</td>
            <td>5 709 586</td>
            <td>93.95</td>
            <td>51 117</td>
            <td>89.28</td>
            <td>17 662</td>
            <td>81.98</td>
        </tr>
        <tr>
            <td>89.2</td>
            <td>4 571 765</td>
            <td>93.72</td>
            <td>40 894</td>
            <td>88.73</td>
            <td>14 130</td>
            <td>80.02</td>
        </tr>
        <tr>
            <td>91.4</td>
            <td>3 661 508</td>
            <td>93.37</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>99.0</td>
            <td>411 432</td>
            <td>91.53</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
    </tbody>
</table>


## The continuation of the study 

title

## Abstract

In this paper, we have researched implementing convolutional neural network (CNN) models for devices with limited resources, such as smartphones and embedded computers. To optimize the number of parameters of these models, we studied various popular methods that would allow them to operate more efficiently. Specifically, our research focused on the ResNet-101 and VGG-19 architectures, which we modified using techniques specific to model optimization. We aimed to determine which approach would work best for particular requirements for a maximum accepted accuracy drop. Our contribution lies in the comprehensive ablation study, which presents the impact of various approaches on the final results, specifically in terms of reducing model parameters, FLOPS, and the potential decline in accuracy. We explored the feasibility of implementing architecture compression methods that can influence the model's structure. Additionally, we delved into post-training methods, such as pruning and quantization, at various model sparsity levels. This study builds upon our prior research in order to provide a more comprehensive understanding of the subject matter at hand.

## Experiments results

### Optimization techniques for ResNet-101 trained on CIFAR-100


|           Approach               | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| -------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|         Fire modules             |  -  |  ✓  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |
|       No FC classifier           |  -  |  -  |  -  |  -  |  -  |  -  |  ✓  |  ✓  |  ✓  |  -  |  ✓  |  ✓  |  ✓  |  ✓  |  -  |
|        Nx1 - 1xN Conv.           |  -  |  -  |  ✓  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |
|            DWconv                |  -  |  -  |  -  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |
|         Shuffle mech.            |  -  |  -  |  -  |  -  |  ✓  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  ✓  |  ✓  |  ✓  |
|       Inv. bottlenecks           |  -  |  -  |  -  |  -  |  -  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |
|      Optimization before GAP     |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  ✓  |  ✓  |  -  |  ✓  |  ✓  |  ✓  |  ✓  |  -  |
| Less channels on the early stage |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  ✓  |  -  |  ✓  |  ✓  |  ✓  |  ✓  |  -  |
|        Ghost bottleneck          |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  -  |
|         SE connections           |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  ✓  |  -  |  ✓  |  ✓  |
|            DiCE unit             |  -  |  ✓  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  ✓  |
|       **Parameters [M]**         |42.697 |59.245 |35.582 |21.425 | 7.721| 2.439| 1.233| 0.919| 0.915| 0.458| 0.408| 1.432| 0.159| 1.182| 3.007|
|       **Parameters [%]**         |100.00 |138.76 | 83.34 | 50.18 | 18.08| 5.71 | 2.89 | 2.15 | 2.14 | 1.07 | 0.96 | 3.35 | 0.37 | 2.77 | 7.04 |
|         **FLOPS [M]**            |2520.41|3480.20|2141.77|1279.72|420.93| 79.39| 63.14| 58.11| 53.75| 32.83| 29.89| 31.61| 12.93| 14.65|160.40|
|         **FLOPS [%]**            |100.00 | 138.08| 84.98 | 50.77 | 16.70| 3.15 | 2.51 | 2.31 | 2.13 | 1.30 | 1.19 | 1.25 | 0.51 | 0.58 | 6.36 |
|         **Size [MB]**            |163.47 | 226.12| 136.34| 82.33 | 30.06| 9.66 | 5.04 | 3.83 | 3.81 | 1.99 | 1.79 | 5.73 | 0.85 | 4.78 | 12.15|
|        **Accuracy [%]**          | 67.51 | 70.81 | 71.19 | 62.80 | 67.32| 67.86| 67.18| 66.52| 66.41| 65.65| 65.49| 66.35| 59.92| 61.45| 63.58|

### Optimization techniques for VGG-19 trained on CIFAR-10


|           Approach          | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
| --------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|             GAP             |  -  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |
|        Nx1 - 1xN Conv.      |  -  |  -  |  ✓  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |
|            DWconv           |  -  |  -  |  -  |  ✓  |  -  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |
|       Fire modules          |  -  |  -  |  -  |  -  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |
|      No FC classifier       |  -  |  -  |  -  |  -  |  -  |  -  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  ✓  |  -  |
|         Shuffle mech.       |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  ✓  |  -  |  ✓  |  ✓  |  ✓  |  ✓  |
|        Ghost modules        |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  ✓  |  ✓  |  ✓  |  -  |  -  |
|         SE connections      |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  ✓  |  ✓  |  ✓  |
|           DiCE unit         |  -  |  -  |  ✓  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  ✓  |
|       **Parameters [M]**    |42.697 |59.245|35.582 |21.425|7.721|2.439|1.233|0.919|0.915|0.458|0.408|1.432|0.159|1.182|3.007|
|       **Parameters [%]**    |100.00 |138.76|83.34  |50.18 |18.08|5.71 |2.89 |2.15 |2.14 |1.07 |0.96 |3.35 |0.37 |2.77 |7.04 |
|         **FLOPS [M]**       |2520.41|3480.20|141.77|1279.72|420.93|79.39|63.14|58.11|53.75|32.83|29.89|31.61|12.93|14.65|160.40|
|         **FLOPS [%]**       |100.00 |138.08| 84.98 |50.77 |16.70| 3.15| 2.51| 2.31| 2.13| 1.30| 1.19| 1.25| 0.51| 0.58| 6.36 |
|         **Size [MB]**       |163.47 |226.12| 136.34| 82.33|30.06| 9.66| 5.04| 3.83| 3.81| 1.99| 1.79| 5.73| 0.85| 4.78| 12.15|
|        **Accuracy [%]**     | 67.51 | 70.81| 71.19 | 62.80|67.32|67.86|67.18|66.52|66.41|65.65|65.49|66.35|59.92|61.45|63.58|


### ResNet-101 trained on CIFAR-100 - pruning

<table>
    <thead>
        <tr>
            <th rowspan=2>Sparsity [%]</th>
            <th colspan=2>no. 1 - baseline</th>
            <th colspan=2>no. 11</th>
            <th colspan=2>no. 13</th>
        </tr>
        <tr>
            <th>Non-zeroed</th>
            <th>Acc [%]</th>
            <th>Non-zeroed</th>
            <th>Acc [%]</th>
            <th>Non-zeroed</th>
            <th>Acc [%]</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>0.0</td>
            <td>42 697 380</td>
            <td>67.51</td>
            <td>408 035</td>
            <td>65.49</td>
            <td>158 815</td>
            <td>59.92</td>
        </tr>
        <tr>
            <td>20.0</td>
            <td>34 114 509</td>
            <td>67.44</td>
            <td>310 870</td>
            <td>65.10</td>
            <td>111 609</td>
            <td>59.37</td>
        </tr>
        <tr>
            <td>35.9</td>
            <td>27 332 567</td>
            <td>66.76</td>
            <td>248 696</td>
            <td>64.35</td>
            <td>89 287</td>
            <td>57.60</td>
        </tr>
        <tr>
            <td>48.7</td>
            <td>21 907 014</td>
            <td>65.73</td>
            <td>198 957</td>
            <td>63.31</td>
            <td>71 430</td>
            <td>54.12</td>
        </tr>
        <tr>
            <td>58.9</td>
            <td>17 566 571</td>
            <td>65.52</td>
            <td>159 166</td>
            <td>61.95</td>
            <td>57 144</td>
            <td>48.56</td>
        </tr>
        <tr>
            <td>67.1</td>
            <td>14 094 217</td>
            <td>64.53</td>
            <td>127 333</td>
            <td>59.15</td>
            <td>45 715</td>
            <td>41.86</td>
        </tr>
        <tr>
            <td>73.7</td>
            <td>11 316 334</td>
            <td>63.37</td>
            <td>101 866</td>
            <td>56.14</td>
            <td>36 572</td>
            <td>35.88</td>
        </tr>
        <tr>
            <td>78.9</td>
            <td>9 094 027</td>
            <td>61.34</td>
            <td>81 493</td>
            <td>52.13</td>
            <td>29 258</td>
            <td>29.64</td>
        </tr>
        <tr>
            <td>83.1</td>
            <td>7 316 182</td>
            <td>59.69</td>
            <td>65 194</td>
            <td>47.47</td>
            <td>23 406</td>
            <td>24.25</td>
        </tr>
        <tr>
            <td>86.5</td>
            <td>5 893 906</td>
            <td>56.39</td>
            <td>52 155</td>
            <td>43.93</td>
            <td>18 725</td>
            <td>18.39</td>
        </tr>
        <tr>
            <td>89.1</td>
            <td>4 756 085</td>
            <td>53.46</td>
            <td>41 724</td>
            <td>39.21</td>
            <td>14 980</td>
            <td>13.14</td>
        </tr>
    </tbody>
</table>
