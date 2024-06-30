# Inject

## Abstract
A CLIP-adapter model for few-shot domain adaptation. The method is in the same spirit as CLIP-adapter https://arxiv.org/pdf/2110.04544.
This project has been open-sourced for showcasing, but is not finished yet.
It builds upon
- CLIP: https://arxiv.org/abs/2103.00020
- CLIP-adapter: https://arxiv.org/pdf/2110.04544
- Prototypical Networks: https://arxiv.org/abs/1703.05175
- DINOv2: https://arxiv.org/abs/2304.07193

## Introduction
We introduce learnable proto-prompts which are learnable convex-combinations of the embeddings for a given set of prompts.
Furthermore, we introduce the ROSE-layer. It first rotates the spherical embeddings of CLIP, then applies squeeze-excitation and finally rotates back to the original space.
This method performs well on all common benchmark datasets with its default hyperparameters and does not tend to overfit and does not rely on a cache.

## Preliminary Results (16-shot)
Resnet50 is used as the CLIP backbone and vit-s as the vision transformer for dinov2.
<table>
  <tr>
    <th>Method</th>
    <th>Standford Cars</th>
    <th>UCF-101</th>
    <th>Caltech-101</th>
    <th>Flower-102</th>
    <th>SUN397</th>
    <th>DTD</th>
    <th>EuroSat</th>
    <th>FGVCAircraft</th>
    <th>OxfordPets</th>
    <th>Food101</th>
  </tr>
  <tr>
    <td>zero-shot (Read of from tables in https://arxiv.org/pdf/2303.02151)
    <td>55.0</td>
    <td>61.0</td>
    <td>86.5</td>
    <td>66.0</td>
    <td>58.5</td>
    <td>42.5</td>
    <td>38.5</td>
    <td>17.5</td>
    <td>85.9</td>
    <td>77.5</td>
  </tr>
  <tr>
    <td>Coop (Read of from tables in https://arxiv.org/pdf/2303.02151)
    <td>73.5</td>
    <td>75.5</td>
    <td>92.0</td>
    <td>94.5</td>
    <td>69.5</td>
    <td>63.5</td>
    <td>83.0</td>
    <td>32.0</td>
    <td>87.0</td>
    <td>75.0</td>
  </tr>
  <tr>
    <td>Clip-Adapter (Caution: Read of from tables in https://arxiv.org/pdf/2303.02151)
    <td>74.0</td>
    <td>77.0</td>
    <td>92.5</td>
    <td>94.5</td>
    <td>69.5</td>
    <td>65.5</td>
    <td>83.0</td>
    <td>32.0</td>
    <td>88.0</td>
    <td>78.1</td>
  </tr>
  <tr>
    <td>Tip-Adapter-F (Cache-based, Caution: Read of from tables in https://arxiv.org/pdf/2303.02151)
    <td>75.5</td>
    <td>77.5</td>
    <td>93.0</td>
    <td>94.5</td>
    <td>71</td>
    <td>65.5</td>
    <td>83.0</td>
    <td>38.0</td>
    <td>89.5</td>
    <td>78.1</td>
  </tr>
  <tr>
    <td>INJECT (No Cache)
    <td>73.8</td>
    <td>78.4</td>
    <td>93.9</td>
    <td>95.6</td>
    <td>not computed yet</td>
    <td>66.5</td>
    <td>84.0</td>
    <td>36.4</td>
    <td>90.5</td>
    <td>79.5</td>
  </tr>
</table>

## Preliminary Results (4-shot)
<table>
  <tr>
    <th>Method</th>
    <th>Standford Cars</th>
    <th>UCF-101</th>
    <th>Caltech-101</th>
    <th>Flower-102</th>
    <th>SUN397</th>
    <th>DTD</th>
    <th>EuroSat</th>
    <th>FGVCAircraft</th>
    <th>OxfordPets</th>
    <th>Food101</th>
  </tr>
  <tr>
    <td>zero-shot (Read of from tables in https://arxiv.org/pdf/2303.02151)
    <td>55.0</td>
    <td>61.0</td>
    <td>86.5</td>
    <td>66.0</td>
    <td>58.5</td>
    <td>42.5</td>
    <td>38.5</td>
    <td>17.5</td>
    <td>85.9</td>
    <td>77.5</td>
  </tr>
  <tr>
    <td>Tip-Adapter-F (Cache-based, Caution: Read of from tables in https://arxiv.org/pdf/2303.02151)
    <td>65.0</td>
    <td>71.2</td>
    <td>91.8</td>
    <td>87.5</td>
    <td>71</td>
    <td>55.6</td>
    <td>74.0</td>
    <td>26.5</td>
    <td>88.3</td>
    <td>77.6</td>
  </tr>
  <tr>
    <td>INJECT (No Cache)
    <td>64.5</td>
    <td>71.3</td>
    <td>91.9</td>
    <td>89.6</td>
    <td>not computed yet</td>
    <td>59.8</td>
    <td>77.6</td>
    <td>26.1</td>
    <td>88.9</td>
    <td>78.5</td>
  </tr>
</table>
The exact results have to be requested from the authors.

## dinov2 16-shot
<table>
  <tr>
    <th>Method</th>
    <th>Standford Cars</th>
    <th>UCF-101</th>
    <th>Caltech-101</th>
    <th>Flower-102</th>
    <th>SUN397</th>
    <th>DTD</th>
    <th>EuroSat</th>
    <th>FGVCAircraft</th>
    <th>OxfordPets</th>
    <th>Food101</th>
  </tr>
  <tr>
    <td>dinov2 16-shot KNN (with augmentation)
    <td>28.0</td>
    <td>67.4</td>
    <td>92.9</td>
    <td>99.1</td>
    <td>not computed yet</td>
    <td>63.8</td>
    <td>63.1</td>
    <td>26.4</td>
    <td>86.5</td>
    <td>61.9</td>
  </tr>
  <tr>
    <td>INJECT
    <td>71.4</td>
    <td>79.2</td>
    <td>96.2</td>
    <td>99.8</td>
    <td>not computed yet</td>
    <td>72.0</td>
    <td>84.4</td>
    <td>59.0</td>
    <td>92.3</td>
    <td>74.3</td>
  </tr>
</table>