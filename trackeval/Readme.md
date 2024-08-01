# 修改mot官网指标测试代码

我是以mot17数据测试作为基准修改的，其余不相干脚本，我将删除，或相关较麻烦的脚本，我已修改。


##指标数据测试结构
###mot17 gt.txt数据格式：
frame, id, bb_left, bb_top, bb_width, bb_height, active, label_id, visibility ratio 

第0个代表第几帧;
第1个值为目标运动轨迹的ID号;
第2个到第5个数代表物体框的左上角坐标及长宽;
第6个值为目标轨迹是否进入考虑范围内的标志，0表示忽略，1表示active;
第7个值为该轨迹对应的目标种类（种类见下面的表格中的label-ID对应情况）;
第8个值为box的visibility ratio，表示目标运动时被其他目标box包含/覆盖或者目标之间box边缘裁剪情况。

特别说明：
第6个activate若为0表示该行目标不考虑计算。
第7个目标种类，目标名称和数字一定要对齐，依照数字指定为目标选择，如下:
"class_name_to_class_id": 
{'pedestrian': 1, 'person_on_vehicle': 2, 'car': 3, 'bicycle': 4, 'motorbike': 5,'non_mot_vehicle': 6, 
'static_person': 7, 'distractor': 8, 'occluder': 9, 'occluder_on_ground': 10, 'occluder_full': 11, 
'reflection': 12, 'crowd': 13}
具体解释如图：
![](trackeval/data/images/mot-format.png)

###预测数据结构

frame_id, id, bbox_left, bbox_top, bbox_w, bbox_h, -1, -1, -1, i
第0个的frame_id表示第几帧，没有0帧，只有第1帧开始，与gt.txt的帧对应;
第1个id表示跟踪id,由跟踪算法决定，实际为常说的track_id;
第2个到第5个数代表物体框的左上角坐标及长宽;
第9个i表示第frame_idx+1帧第几个目标id，也可以固定为-1，该参数不需要。

特别说明：
指标预测模型是第一帧到最后一帧by顺序一张一张图预测；
输出结果为预测txt保存结果，名称需和seq相同，如MOT17-02-FRCNN.txt


![](trackeval/data/images/mot-predect-format.png)

##文件结构
###gt文件夹:
主路径下是被测文件夹列表，每个列表文件(如：MOT17-02-FRCNN)下有一个gt.txt文件和seqinfo.ini文件，
该路径可由config['GT_LOC_FORMAT'] = '{gt_folder}/{seq}/gt/gt.txt'参数控制。
其中seqinfo.ini文件用于记录name=MOT17-02-FRCNN文件下相关信息，主要使用seqLength信息。
seqinfo.ini如下:

'''

    [Sequence]
    name=MOT17-02-FRCNN
    imDir=img1
    frameRate=30
    seqLength=600
    imWidth=1920
    imHeight=1080
    imExt=.jpg

'''
也可变为如下：
'''

    [Sequence]
    seqLength=600

'''


###predect文件夹:
主路径下只有对应txt文件，命名分别为gt文件对应的文件列表名称，MOT17-02-FRCNN.txt
具体方法如下图：
![](trackeval/data/images/file-format.png)



##运行参数说明
在mot_challenge_2d_box.py文件夹下有类MotChallenge2DBox(_BaseDataset)，用于处理相关数据，
而在该类中有一个默认default_config字典，我已做了修改，具体用法如下解释。
'''

    default_config = {
            'GT_FOLDER': os.path.join(code_path, 'data/'),  # 真实标签路径Location of GT data
            'TRACKERS_FOLDER': os.path.join(code_path, 'data/predect_mot/'),  # 预测标签路径 Trackers location
            'OUTPUT_FOLDER': None,  # 指标保存路径，若为None将保存在TRACKERS_FOLDER文件下
            # 'TRACKERS_TO_EVAL': None,  # Filenames of predect_mot to eval (if None, all in folder)
            'CLASSES_TO_EVAL': ['pedestrian'],  # 规定哪些类别预测指标 Valid: ['pedestrian']
            'BENCHMARK': 'MOT17',  # 用于显示名字，随便起名，我是以mot17为基准修改，因此默认为MOT17

            'PRINT_CONFIG': True,  # Whether to print current config
            'DO_PREPROC': True,  # Whether to perform preprocessing (never done for MOT15)
            'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER

            'TRACKER_DISPLAY_NAMES': None,  # Names of predect_mot to display, if None: TRACKERS_TO_EVAL
            'SEQMAP_FOLDER': None,  # Where seqmaps are found (if None, GT_FOLDER/seqmaps)
            'SEQMAP_FILE': None,  # Directly specify seqmap file (if none use seqmap_folder/benchmark-split_to_eval)
            'SEQ_INFO': None,  # 需要预测的列表，若为None则将GT_FOLDER文件均当做seq预测指标

            'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',  # 能找到真实标签txt路径,gt_folder为GT_FOLDER，seq为seq_list遍历值
            'SKIP_SPLIT_FOL': False,  # If False, data is in GT_FOLDER/BENCHMARK-SPLIT_TO_EVAL/ and in
                                      # TRACKERS_FOLDER/BENCHMARK-SPLIT_TO_EVAL/tracker/
                                      # If True, then the middle 'benchmark-split' folder is skipped for both.
            "class_name_to_class_id": {'pedestrian': 1, 'person_on_vehicle': 2, 'car': 3, 'bicycle': 4, 'motorbike': 5,
                                      'non_mot_vehicle': 6, 'static_person': 7, 'distractor': 8, 'occluder': 9,
                                      'occluder_on_ground': 10, 'occluder_full': 11, 'reflection': 12, 'crowd': 13},
            'use_super_categories':False, # 决定是否合并类别为key类预测
            'super_categories' : {"FF": ['pedestrian', 'car']}  # 表示将字典value类统一到key上，给出指标结果
        }


'''
###参数重点说明
①合并类指标测试方法
use_super_categories:参数为True将会将我们在super_categories指定合并类一起测试
'super_categories' :指定合并类，如将'pedestrian', 'car'合并为’FF‘，如: {"FF": ['pedestrian', 'car']}

②测试类指定方法
CLASSES_TO_EVAL:列表，指定gt.txt需要测试的类别，特别说明，它和gt.txt的第6个activate共同决定有效gt目标
③指定gt路径方法
GT_LOC_FORMAT:如{gt_folder}/{seq}/gt/gt.txt，确定gt路径
④指定测试方法
SEQ_INFO:指定gt_folder测试方法
⑤确定gt类别方法
class_name_to_class_id:指定名称和类别对应字典，gt.txt中的label id为对应数字，此为②提供依据


##自定义参数

通过config字典，修改MotChallenge2DBox(_BaseDataset)中的默认参数，如下:

'''

    config={}
    config['TRACKERS_FOLDER'] = ROOT+'/data/predect_mot' # 预测路径
    config['GT_FOLDER'] = ROOT+'/data/mot17_gt'   # 给出gt路径
    config['OUTPUT_FOLDER'] = ROOT+'/data/out_dir'
    # 确定文件内gt.txt的路径，gt_folder=config['GT_FOLDER']，seq为os.listdir(gt_folder)列表
    config['GT_LOC_FORMAT'] = '{gt_folder}/{seq}/gt/gt.txt'
    config['CLASSES_TO_EVAL'] = ['pedestrian']  # 确定预测指标的类别
    dataset = MotChallenge2DBox(config)  # dataset_list是存放数据信息列表


'''



##运行命令

按照以上参数修改，直接运行run_mot_challenge.py文件

#tools文件工具
mot_vision.py文件用于将gt.txt信息可视化图像上，效果如下图:

![](trackeval/tools/img.png)


##测试结果

![](data/images/result.png)


# TrackEval
*Code for evaluating object tracking.*

This codebase provides code for a number of different tracking evaluation metrics (including the [HOTA metrics](https://link.springer.com/article/10.1007/s11263-020-01375-2)), as well as supporting running all of these metrics on a number of different tracking benchmarks. Plus plotting of results and other things one may want to do for tracking evaluation.

## **NEW**: RobMOTS Challenge 2021

Call for submission to our [RobMOTS Challenge](https://eval.vision.rwth-aachen.de/rvsu-workshop21/?page_id=110) (Robust Multi-Object Tracking and Segmentation) held in conjunction with our [RVSU CVPR'21 Workshop](https://eval.vision.rwth-aachen.de/rvsu-workshop21/). Robust tracking evaluation against 8 tracking benchmarks. Challenge submission deadline June 15th. Also check out our workshop [call for papers](https://eval.vision.rwth-aachen.de/rvsu-workshop21/?page_id=74).

## Official Evaluation Code

The following benchmarks use TrackEval as their official evaluation code, check out the links to see TrackEval in action:

 - **[RobMOTS](https://eval.vision.rwth-aachen.de/rvsu-workshop21/?page_id=110)** ([Official Readme](docs/RobMOTS-Official/Readme.md))
 - **[KITTI Tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php)**
 - **[KITTI MOTS](http://www.cvlibs.net/datasets/kitti/eval_mots.php)**
 - **[MOTChallenge](https://motchallenge.net/)** ([Official Readme](docs/MOTChallenge-Official/Readme.md))
 - **[Open World Tracking](https://openworldtracking.github.io)** ([Official Readme](docs/OpenWorldTracking-Official))
 - **[PersonPath22](https://amazon-research.github.io/tracking-dataset/personpath22.html)**
 <!--- **[MOTS-Challenge](https://motchallenge.net/data/MOTS/)** ([Official Readme](docs/MOTS-Challenge-Official/Readme.md)) --->

If you run a tracking benchmark and want to use TrackEval as your official evaluation code, please contact Jonathon (contact details below).

## Currently implemented metrics

The following metrics are currently implemented:

Metric Family | Sub metrics | Paper | Code | Notes |
|----- | ----------- |----- | ----------- | ----- |
| | | |  |  |
|**HOTA metrics**|HOTA, DetA, AssA, LocA, DetPr, DetRe, AssPr, AssRe|[paper](https://link.springer.com/article/10.1007/s11263-020-01375-2)|[code](trackeval/metrics/hota.py)|**Recommended tracking metric**|
|**CLEARMOT metrics**|MOTA, MOTP, MT, ML, Frag, etc.|[paper](https://link.springer.com/article/10.1155/2008/246309)|[code](trackeval/metrics/clear.py)| |
|**Identity metrics**|IDF1, IDP, IDR|[paper](https://arxiv.org/abs/1609.01775)|[code](trackeval/metrics/identity.py)| |
|**VACE metrics**|ATA, SFDA|[paper](https://link.springer.com/chapter/10.1007/11612704_16)|[code](trackeval/metrics/vace.py)| |
|**Track mAP metrics**|Track mAP|[paper](https://arxiv.org/abs/1905.04804)|[code](trackeval/metrics/track_map.py)|Requires confidence scores|
|**J & F metrics**|J&F, J, F|[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Perazzi_A_Benchmark_Dataset_CVPR_2016_paper.pdf)|[code](trackeval/metrics/j_and_f.py)|Only for Seg Masks|
|**ID Euclidean**|ID Euclidean|[paper](https://arxiv.org/pdf/2103.13516.pdf)|[code](trackeval/metrics/ideucl.py)| |


## Currently implemented benchmarks

The following benchmarks are currently implemented:

Benchmark | Sub-benchmarks | Type | Website | Code | Data Format |
|----- | ----------- |----- | ----------- | ----- | ----- |
| | | |  |  | |
|**RobMOTS**|Combination of 8 benchmarks|Seg Masks|[website](https://eval.vision.rwth-aachen.de/rvsu-workshop21/?page_id=110)|[code](trackeval/datasets/rob_mots.py)|[format](docs/RobMOTS-Official/Readme.md)|
|**Open World Tracking**|TAO-OW|OpenWorld / Seg Masks|[website](https://openworldtracking.github.io)|[code](trackeval/datasets/tao_ow.py)|[format](docs/OpenWorldTracking-Official/Readme.md)|
|**MOTChallenge**|MOT15/16/17/20|2D BBox|[website](https://motchallenge.net/)|[code](trackeval/datasets/mot_challenge_2d_box.py)|[format](docs/MOTChallenge-format.txt)|
|**KITTI Tracking**| |2D BBox|[website](http://www.cvlibs.net/datasets/kitti/eval_tracking.php)|[code](trackeval/datasets/kitti_2d_box.py)|[format](docs/KITTI-format.txt)|
|**BDD-100k**| |2D BBox|[website](https://bdd-data.berkeley.edu/)|[code](trackeval/datasets/bdd100k.py)|[format](docs/BDD100k-format.txt)|
|**TAO**| |2D BBox|[website](https://taodataset.org/)|[code](trackeval/datasets/tao.py)|[format](docs/TAO-format.txt)|
|**MOTS**|KITTI-MOTS, MOTS-Challenge|Seg Mask|[website](https://www.vision.rwth-aachen.de/page/mots)|[code](trackeval/datasets/mots_challenge.py) and [code](trackeval/datasets/kitti_mots.py)|[format](docs/MOTS-format.txt)|
|**DAVIS**|Unsupervised|Seg Mask|[website](https://davischallenge.org/)|[code](trackeval/datasets/davis.py)|[format](docs/DAVIS-format.txt)|
|**YouTube-VIS**| |Seg Mask|[website](https://youtube-vos.org/dataset/vis/)|[code](trackeval/datasets/youtube_vis.py)|[format](docs/YouTube-VIS-format.txt)|
|**Head Tracking Challenge**| |2D BBox|[website](https://arxiv.org/pdf/2103.13516.pdf)|[code](trackeval/datasets/head_tracking_challenge.py)|[format](docs/MOTChallenge-format.txt)|
|**PersonPath22**| |2D BBox|[website](https://github.com/amazon-research/tracking-dataset)|[code](trackeval/datasets/person_path_22.py)|[format](docs/MOTChallenge-format.txt)|
|**BURST**| {Common, Long-tail, Open-world} Class-guided, {Point, Box, Mask} Exemplar-guided |Seg Mask|[website](https://github.com/Ali2500/BURST-benchmark)|[format](https://github.com/Ali2500/BURST-benchmark/blob/main/ANNOTATION_FORMAT.md)|

## HOTA metrics

This code is also the official reference implementation for the HOTA metrics:

*[HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking](https://link.springer.com/article/10.1007/s11263-020-01375-2). IJCV 2020. Jonathon Luiten, Aljosa Osep, Patrick Dendorfer, Philip Torr, Andreas Geiger, Laura Leal-Taixe and Bastian Leibe.*

HOTA is a novel set of MOT evaluation metrics which enable better understanding of tracking behavior than previous metrics.

For more information check out the following links:
 - [Short blog post on HOTA](https://jonathonluiten.medium.com/how-to-evaluate-tracking-with-the-hota-metrics-754036d183e1) - **HIGHLY RECOMMENDED READING**
 - [IJCV version of paper](https://link.springer.com/article/10.1007/s11263-020-01375-2) (Open Access)
 - [ArXiv version of paper](https://arxiv.org/abs/2009.07736)
 - [Code](trackeval/metrics/hota.py)

## Properties of this codebase

The code is written 100% in python with only numpy and scipy as minimum requirements.

The code is designed to be easily understandable and easily extendable. 

The code is also extremely fast, running at more than 10x the speed of the both [MOTChallengeEvalKit](https://github.com/dendorferpatrick/MOTChallengeEvalKit), and [py-motmetrics](https://github.com/cheind/py-motmetrics) (see detailed speed comparison below).

The implementation of CLEARMOT and ID metrics aligns perfectly with the [MOTChallengeEvalKit](https://github.com/dendorferpatrick/MOTChallengeEvalKit).

By default the code prints results to the screen, saves results out as both a summary txt file and a detailed results csv file, and outputs plots of the results. All outputs are by default saved to the 'tracker' folder for each tracker.

## Running the code

The code can be run in one of two ways:

 - From the terminal via one of the scripts [here](scripts/). See each script for instructions and arguments, hopefully this is self-explanatory.
 - Directly by importing this package into your code, see the same scripts above for how. 

## Quickly evaluate on supported benchmarks

To enable you to use TrackEval for evaluation as quickly and easily as possible, we provide ground-truth data, meta-data and example trackers for all currently supported benchmarks.
You can download this here: [data.zip](https://omnomnom.vision.rwth-aachen.de/data/TrackEval/data.zip) (~150mb).

The data for RobMOTS is separate and can be found here: [rob_mots_train_data.zip](https://omnomnom.vision.rwth-aachen.de/data/RobMOTS/train_data.zip) (~750mb).

The data for PersonPath22 is separate and can be found here: [person_path_22_data.zip](https://tracking-dataset-eccv-2022.s3.us-east-2.amazonaws.com/person_path_22_data.zip) (~3mb).

The easiest way to begin is to extract this zip into the repository root folder such that the file paths look like: TrackEval/data/gt/...

This then corresponds to the default paths in the code. You can now run each of the scripts [here](scripts/) without providing any arguments and they will by default evaluate all trackers present in the supplied file structure. To evaluate your own tracking results, simply copy your files as a new tracker folder into the file structure at the same level as the example trackers (MPNTrack, CIWT, track_rcnn, qdtrack, ags, Tracktor++, STEm_Seg), ensuring the same file structure for your trackers as in the example.

Of course, if your ground-truth and tracker files are located somewhere else you can simply use the script arguments to point the code toward your data.

To ensure your tracker outputs data in the correct format, check out our format guides for each of the supported benchmarks [here](docs), or check out the example trackers provided.

## Evaluate on your own custom benchmark

To evaluate on your own data, you have two options:
 - Write custom dataset code (more effort, rarely worth it).
 - Convert your current dataset and trackers to the same format of an already implemented benchmark.

To convert formats, check out the format specifications defined [here](docs).

By default, we would recommend the MOTChallenge format, although any implemented format should work. Note that for many cases you will want to use the argument ```--DO_PREPROC False``` unless you want to run preprocessing to remove distractor objects.

## Requirements
 Code tested on Python 3.7.
 
 - Minimum requirements: numpy, scipy
 - For plotting: matplotlib
 - For segmentation datasets (KITTI MOTS, MOTS-Challenge, DAVIS, YouTube-VIS): pycocotools
 - For DAVIS dataset: Pillow
 - For J & F metric: opencv_python, scikit_image
 - For simples test-cases for metrics: pytest

use ```pip3 -r install requirements.txt``` to install all possible requirements.

use ```pip3 -r install minimum_requirments.txt``` to only install the minimum if you don't need the extra functionality as listed above.

## Timing analysis

Evaluating CLEAR + ID metrics on Lift_T tracker on MOT17-train (seconds) on a i7-9700K CPU with 8 physical cores (median of 3 runs):		
Num Cores|TrackEval|MOTChallenge|Speedup vs MOTChallenge|py-motmetrics|Speedup vs py-motmetrics
:---|:---|:---|:---|:---|:---
1|9.64|66.23|6.87x|99.65|10.34x
4|3.01|29.42|9.77x| |33.11x*
8|1.62|29.51|18.22x| |61.51x*

*using a different number of cores as py-motmetrics doesn't allow multiprocessing.
				
```
python scripts/run_mot_challenge.py --BENCHMARK MOT17 --TRACKERS_TO_EVAL Lif_T --METRICS CLEAR Identity --USE_PARALLEL False --NUM_PARALLEL_CORES 1  
```
				
Evaluating CLEAR + ID metrics on LPC_MOT tracker on MOT20-train (seconds) on a i7-9700K CPU with 8 physical cores (median of 3 runs):	
Num Cores|TrackEval|MOTChallenge|Speedup vs MOTChallenge|py-motmetrics|Speedup vs py-motmetrics
:---|:---|:---|:---|:---|:---
1|18.63|105.3|5.65x|175.17|9.40x

```
python scripts/run_mot_challenge.py --BENCHMARK MOT20 --TRACKERS_TO_EVAL LPC_MOT --METRICS CLEAR Identity --USE_PARALLEL False --NUM_PARALLEL_CORES 1
```

## License

TrackEval is released under the [MIT License](LICENSE).

## Contact

If you encounter any problems with the code, please contact [Jonathon Luiten](https://www.vision.rwth-aachen.de/person/216/) ([luiten@vision.rwth-aachen.de](mailto:luiten@vision.rwth-aachen.de)).
If anything is unclear, or hard to use, please leave a comment either via email or as an issue and I would love to help.

## Dedication

This codebase was built for you, in order to make your life easier! For anyone doing research on tracking or using trackers, please don't hesitate to reach out with any comments or suggestions on how things could be improved.

## Contributing

We welcome contributions of new metrics and new supported benchmarks. Also any other new features or code improvements. Send a PR, an email, or open an issue detailing what you'd like to add/change to begin a conversation.

## Citing TrackEval

If you use this code in your research, please use the following BibTeX entry:

```BibTeX
@misc{luiten2020trackeval,
  author =       {Jonathon Luiten, Arne Hoffhues},
  title =        {TrackEval},
  howpublished = {\url{https://github.com/JonathonLuiten/TrackEval}},
  year =         {2020}
}
```

Furthermore, if you use the HOTA metrics, please cite the following paper:

```
@article{luiten2020IJCV,
  title={HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking},
  author={Luiten, Jonathon and Osep, Aljosa and Dendorfer, Patrick and Torr, Philip and Geiger, Andreas and Leal-Taix{\'e}, Laura and Leibe, Bastian},
  journal={International Journal of Computer Vision},
  pages={1--31},
  year={2020},
  publisher={Springer}
}
```

If you use any other metrics please also cite the relevant papers, and don't forget to cite each of the benchmarks you evaluate on.
