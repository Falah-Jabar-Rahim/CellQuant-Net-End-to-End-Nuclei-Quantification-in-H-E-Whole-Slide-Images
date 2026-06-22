# CellQuant-Net
![WSI-QA](./Figs/Fig1.png)








































<p align="justify">  </p>

Check out the paper: [Paper](https://www.sciencedirect.com/science/article/pii/S266652122500122X?via%3Dihub)

# Setting Up the Pipeline:

1. System requirements:
- Ubuntu 20.04 or 22.04
- CUDA version: 12.8
- Python version: 3.9 (using conda environments)
- Anaconda version 23.7.4
2. Steps to Set Up the Pipeline:
- `cd ~/Desktop`
- `git clone https://github.com/Falah-Jabar-Rahim/CellQuant-Net-End-to-End-Nuclei-Quantification-in-H-E-Whole-Slide-Images.git CellQuant-Net`
- `cd CellQuant-Net`
- `chmod +x install.sh`
- `./install.sh`
- `conda activate cellquantnet`
- `python verify_installation.py`

# Demo:

- Downlaod: [test WSI](https://portal.gdc.cancer.gov/files/5bd34bab-6a75-4d62-ab9d-0ada84414776)
- put the WSI in the input folder
- run the command `python run_cellquant_net.py  --cpu_workers 32  --batch_size 128 --cell_connectivity   --model_type tnmi_20x.pth`





  
# Datasets:

- Contact the corresponding author to access the datasets described in the paper
- The datasets (only tiles and segmentation masks) are available for research purposes onlypython run_cellquant_net.py  --cpu_workers 32  --batch_size 128 --cell_connectivity   --model_type tnmi_20x.pth


# Pretrained Weights:

- Download the pretrained weights from this link: [Model weights](https://huggingface.co/jwtyar/WSI-SmartTiling/tree/main)
- Place the artifact detection weights in the `pretrained_ckpt` folder, and the pen-marking detection and removal weights in the `/Ink_Removal/pre-trained` folder


# Inference:

<p align="justify"> The pipeline starts by identifying the WSI tissue region and dividing it into smaller image tiles (e.g., 270x270). Pen-marking detection is then applied to categorize the tiles into two classes: those with high pen-marking (which are discarded) and those with medium and low pen-marking. Tiles with medium and low pen-marking undergo a pen-marking removal process, resulting in clean image tiles. Next, the clean image tiles are fed into the proposed artifact detection model to identify artifacts, followed by an optimization technique to select the best tiles—those with minimal artifacts and background and maximum qualified tissue. Finally, the WSI is reconstructed by combining the selected tiles to generate the final output. Additionally, the model generates a segmentation for the entire WSI and also provides statistics on the tile segmentations. </p>

- Place your Whole Slide Image (WSI) into the `test_wsi` folder
- The pre-trained weights for artifact detection are available in the `pretrained_ckpt` folder, while the weights for pen-marker removal are located in the `Ink_Removal/pre-trained` folder
- In the terminal execute:
  `python test_wsi.py`

- After running the inference, you will obtain the following outputs in `test_wsi` folder:
    - A thumbnail image of WSI
    - A thumbnail image of WSI with regions of interest (ROI) identified
    - A segmentation mask highlighting segmented regions of the WSI [Qualifed tissue: green, fold: red, blur: orange, and background: black]
    - A segmentation mask highlighting only qualified tissue regions of the WSI [background:0, qualified tissue:1]
    - Excel files contain statistics on identified artifacts
    - A folder named Selected_tiles containing qualified tiles

  
# Important Notes:

- If your WSIs do not contain pen-marking artifacts, use the faster pipeline available here: [GitHub](https://github.com/Falah-Jabar-Rahim/A-Fully-Automatic-DL-Pipeline-for-WSI-QA)
- Use the parameter `wsilevel` in `configs/inti_artifact.py` to specify the WSI level used for tiling
	- wsilevel = 0 tiles the WSI at native resolution, providing maximum segmentation performance, but with longer inference time. Recommended when preparing high-quality datasets for training deep learning models
	- wsilevel = 1 or 2 tiles the WSI at lower magnification, resulting in faster inference, but with reduced segmentation accuracy. This may be sufficient when preparing datasets for downstream tasks or testing scenarios where speed is prioritized over accuracy
- If your WSI image has a format other than .svs or .mrxs, please modify line 92 in `test_wsi.py`
- It is recommended to use a tile size of 270 × 270 pixels
- If your WSI image contains pen-markings other than red, blue, green, or black, please update the `pens.py` file (located in the `Ink_Removal/wsi_tile_cleanup/filters folder`) to handle any additional pen-markings
- To generate a high-resolution thumbnail image and segmentation masks, you can adjust the `thumbnail_size` parameter in `configs/inti_artifact.py`. However, note that this will increase the execution time and memory usage
- To generate tiles of different sizes (e.g., 512x512):
    - Run the pipeline to generate the qualified tissue mask (binary mask: tissure (1)/bg or artifatcs (0))
    - Run `python tile_wsi.py --wsi_path input_WSI/normal_3.svs  --mask_path /tissue mask/path/ --out_dir /output dir/path/ --tile_size 512 512 --overlap 0.0 --bg_thr 0.10 -level 0 --save_overlay`
- To disable pen-mark removal model, set the `clean_penmarker` parameter in `configs/inti_artifact.py` to 0. When disabled, any tile with pen-marking greater than `min_marker_thr` will be removed. You can adjust `min_marker_thr` values to achieve better results for your dataset
- These parameters are important and have a significant impact on model performance. Adjust them according to your dataset: line 60 of `configs/inti_artifact.py` `back_thr` , `blur_fold_thr` , `max_marker_thr` , `min_marker_thr` , `clean_penmarker`
- Lower the batch size (`batch_size`in `configs/inti_artifact.py`) if you encounter Out of Memory (OOM) issues

# Training:

- To retrain the artifact detection model, refer to the details provided in: [GitHub](https://github.com/Falah-Jabar-Rahim/A-Fully-Automatic-DL-Pipeline-for-WSI-QA)
- To retrain the ink removal detection model, refer to the details provided in: [GitHub](https://github.com/Vishwesh4/Ink-WSI)

# Results & Benchmarking

![WSI-QA](./Figs/Fig.3.png)
![WSI-QA](./Figs/Fig.4.png)
![WSI-QA](./Figs/Fig.5.png)
![WSI-QA](./Figs/Fig.7.png)

<p align="justify"> Benchmark models, GrandQC (https://github.com/cpath-ukk/grandqc) pixel-wise segmentation model developed for artifact detection, and four tile-wise classification models with different network architectures (https://github.com/NeelKanwal/Equipping-Computational-Pathology-Systems-with-Artifact-Processing-Pipeline), namely MoE-CNN, MoE-ViT, multiclass-CNN, and multiclass-ViT. The proposed pixel-wise segmentation model is compared to GrandQC based on pixel segmentation accuracy and to MoE-CNN, MoE-ViT, Multiclass-CNN, and Multiclass-ViT based on tile classification. The classification considers three classes—artifact-free, fold, and blur. The model takes input tiles and generates segmentation masks, which are then used for tile classification. The classification process follows these criteria: (1) If the background occupies more than 50% of the tile, it is classified as a background tile. (2) If the background occupies less than 50%, but blurring and/or folding artifacts exceed 10% of the tile, it is classified as either fold or blur. (3) If the background is less than 50% and blurring and/or folding artifacts are below 10%, the tile is classified as artifact-free. The internal and external datasets are described in the manuscript. For segmentation, the ground truth segmentation masks are compared to the segmentation masks generated by the model. For classification, the predicted classes are compared to the ground truth labels. Quantitative metrics, including total accuracy (Acc), precision, recall, and F1 score, were used to evaluate classification performance, and the Dice metric was used to evaluate segmentation performance. The source code and model weights for benchmark models were obtained from the original GitHub repositories. The dataset includes tiles and their corresponding segmentation masks, each with a resolution of 270×270 pixels. The input tile size for the proposed segmentation model and MoE-CNN is the same and is resized to 288×288 to ensure compatibility with QrandQC </p>

# Acknowledgment:

Some parts of this pipeline were adapted from work on [GitHub](https://github.com/pengsl-lab/DHUnet) and [GitHub](https://github.com/Vishwesh4/Ink-WSI). If you use this pipeline, please make sure to cite their work properly

# Citation:

Falah Jabar, Lill-Tove Rasmussen Busund, Biagio Ricciuti, Masoud Tafavvoghi, Thomas K. Kilvaer, David J. Pinato, Mette Pøhl, Sigve Andersen, Tom Donnem, David J. Kwiatkowski, Mehrdad Rakaee,
Fully Automatic Content-Aware Tiling Pipeline for Pathology Whole Slide Images,
Intelligence-Based Medicine,
2025,
100318,
ISSN 2666-5212,
https://doi.org/10.1016/j.ibmed.2025.100318.
(https://www.sciencedirect.com/science/article/pii/S266652122500122X)

# Contact:

If you have any questions or comments, please feel free to contact: falah.rahim@unn.no

