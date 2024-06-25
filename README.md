# Inject

## Abstract
A CLIP-adapter model for few-shot domain adaptation.

## Introduction
We introduce learnable proto-prompts which are learnable convex-combinations of the embeddings for a given set of prompts.
Furthermore, we introduce the ROSE-layer. It first rotates the spherical embeddings of CLIP, then applies squeeze-excitation and finally rotates back to the original space.
This method performs well on all common benchmark datasets with its default hyperparameters and does not tend to overfit and does not rely on a cache.

## Dependencies
List all libraries and tools required to run the code. For example:
- Python 3.8
- PyTorch 1.11
- NumPy 1.22

## Preliminary Results (16-shot)
<table>
  <tr>
    <th>Method</th>
    <th>Standford Cars</th>
    <th>UCF-101 (Cache)</th>
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
    <td>91.3</td>
    <td>78.1</td>
  </tr>
  <tr>
    <td>INJECT (only learnable prompt Interpolation)
    <td>56.25</td>
    <td>63.5</td>
    <td>88.9</td>
    <td>69.3</td>
    <td>not computed yet</td>
    <td>43.3</td>
    <td>40.0</td>
    <td>not computed yet</td>
    <td>86.5</td>
    <td>78.2</td>
  </tr>
  <tr>
    <td>INJECT (Best Score)
    <td>72.9</td>
    <td>78.9</td>
    <td>93.3</td>
    <td>94.2</td>
    <td>not computed yet</td>
    <td>66.2</td>
    <td>81.8</td>
    <td>not computed yet</td>
    <td>90.1</td>
    <td>79.6</td>
  </tr>
  <tr>
    <td>INJECT (Default value alpha=.9, no hyperparameter tuning)
    <td>71.9</td>
    <td>78.9</td>
    <td>93.1</td>
    <td>93.21</td>
    <td>not computed yet</td>
    <td>66.2</td>
    <td>81.2</td>
    <td>not computed yet</td>
    <td>89.3</td>
    <td>78.4</td>
  </tr>
</table>
The exact results have to be requested from the authors.