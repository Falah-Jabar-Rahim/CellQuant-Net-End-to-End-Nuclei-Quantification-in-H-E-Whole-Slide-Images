# CellQuant-Net
![WSI-QA](./Figs/Fig1.png)
<p align="justify"> Accurate nuclei detection and classification in hematoxylin and eosin (H&E) whole-slide images (WSIs) is a key task in computational pathology, particularly for quantitative analysis of the tumor microenvironment. However, this task remains highly challenging due to variations in nuclei morphology, staining procedures, scanners, organs, magnifications, and WSI artifacts. In addition, many existing pipelines rely on computationally demanding architectures and post-processing procedures, making gigapixel WSI analysis time-consuming. In this work, CellPriorNet (CP-Net) is proposed, an efficient nuclei detection and classification pipeline that utilizes a lightweight convolutional neural network architecture and hematoxylin (H) channel prior information to enhance nuclei-aware feature learning. Extensive benchmarking was conducted against state-of-the-art pipelines on 8 public and private datasets (total:~10.4M nuclei) obtained from different organs, scanners, magnifications, and clinical centers. Experimental results demonstrate that CP-Net achieves comparable performance while significantly reducing inference time. Furthermore, CellQuant-Net was introduced–an end-to-end nuclei quantification pipeline–that integrates a quality assessment (QA) model to exclude regions with artifacts, followed by CP-Net, and a nuclei connectivity analysis to compute nuclei quantitative metrics. </p>

Check out the paper: [Paper]

# Setting Up the Pipeline:

1. System requirements:
- Ubuntu 20.04 or 22.04
- CUDA version: 12.8
- Python version: 3.9 (using conda environments)
- Anaconda version 23.7.4
2. Steps to Set Up the Pipeline:
- Open terminal 
- `cd ~/Desktop`
- `git clone https://github.com/Falah-Jabar-Rahim/CellQuant-Net.git CellQuant-Net`
- `cd CellQuant-Net`
- `chmod +x install.sh`
- `./install.sh`
- `conda activate cellquantnet`
- `python verify_installation.py`

# Quick Test:
- Download a [test WSI](https://portal.gdc.cancer.gov/files/5bd34bab-6a75-4d62-ab9d-0ada84414776)
- Place the downloaded WSI in the `input` folder
- Obtain the pretrained model weights for WSI-QA and CP-Net, then place the CP-Net weights in `CP-Net/weights/` and the WSI-QA weights in `WSI_QA/pretrained_ckpt/`
- Run CellQuant-Net: `python run_cellquant_net.py  --cpu_workers 32  --batch_size 128 --cell_connectivity   --model_type PanNuke.pth`
- After running CellQuant-Net, all results are saved in the `output` folder


# 🎥 Watch the Demo:
[![Installation Tutorial](Figs/Demo.png)](https://youtu.be/RhCJnUfuYkA?is=Jc4keTUtecEcjeZd)


# Pre-trained Models
| Model | Type | Magnification |
|----------|--------|---------------|
| PanNuke.pth | Multi-organ |  40× |
| CoNSeP.pth | Single-organ (Colon) |  40× |
| ILCD.pth | Single-organ (Liver) |  40× |
| Lizard.pth | Single-organ (Colon) |  20× |
| NuCLS.pth | Single-organ (Breast) |  20× |
| PanopTILs.pth | Single-organ (Breast) |  40× |
| SegPath.pth | Multi-organ |  40× |
| TNMI20x.pth | Single-organ (Lung) |  20× |
| TNMI40x.pth | Single-organ (Lung)|  40× |

CP-Net was trained on multiple public and private datasets to support diverse downstream digital pathology applications. Contact the corresponding author to request access to the pre-trained models for WSI-QA and CP-Net models.

# Output Structure
Place the WSIs in the input folder. Download the pretrained model weights for WSI-QA and CP-Net, then place the CP-Net weights in CP-Net/weights/ and the WSI-QA weights in WSI_QA/pretrained_ckpt/. After running CellQuant-Net, all results are saved in the output directory:

### Quality Assessment (QA)
```text
output/
└── QA/
    ├── WSI1/
    ├── WSI2/
    ├── ...
    └── WSI_Summary.xlsx
```
For each WSI, the QA module generates:
| File/Folder | Description |
|-------------|-------------|
| `Qualified/` | High-quality image tiles selected for analysis. |
| `Qualified_H/` | Hematoxylin (H-channel) version of the qualified tiles. |
| `Unqualified/` | Tiles excluded due to excessive background, blur, folds, or other artifacts. |
| `*_seg.png` | WSI-level tissue and artifact segmentation map. |
| `*_thumbnail.png` | WSI thumbnail image. |
| `*_thumbnail_roi.png` | Thumbnail highlighting the selected tissue regions. |
| `*_stats.xlsx` | Tile-level and WSI-level quality assessment statistics. |
| `WSI_Summary.xlsx` | Summary statistics for all processed WSIs. |

### Cell Detection and Classification (CP-Net)
```text
output/
└── CP_Net/
    ├── WSI1/
    ├── WSI2/
    ├── ...
```

For each WSI, the CP-Net module generates:

| File/Folder | Description |
|-------------|-------------|
| `qupath_cells.geojson` | Detected cells exported in QuPath-compatible GeoJSON format. |
| `cell_type_stats.csv` | Cell counts and percentages for each cell type. |
| `cell_neighborhood/` | Cell spatial connectivity analysis results. |
| `connectivity_edges.csv` | Cell-to-cell connectivity graph edges. |
| `qupath_cells_connectivity.geojson` | Connectivity graph exported for visualization in QuPath. |

# Qupath Visualization 
- Open QuPath and load the WSI that was analyzed by the pipeline.
- Open the corresponding `output` folder generated by CP-Net.
- Drag and drop `qupath_cells.geojson` onto the WSI viewer to visualize detected and classified cells.
- Drag and drop `qupath_cells_connectivity.geojson` onto the WSI viewer to visualize the nuclei connectivity network.

# Acknowledgment:

Some parts of this pipeline were adapted from work on [GitHub](https://github.com/DingXiaoH/RepLKNet-pytorch). If you use this pipeline, please make sure to cite their work properly

# Citation:





# Contact:

If you have any questions or comments, please feel free to contact: falah.rahim@unn.no

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Note: This pipeline is intended for research and AI-assisted analysis only. At this stage, the AI model provides predictions and analytical outputs, not clinical decisions.
