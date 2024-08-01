import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import cv2


img_format=['.jpg','.png','.bmp','.PNG','.JPG']

def build_dir(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir,exist_ok=True)
    return out_dir


def build_files(root):
    '''
    :得到该路径下的所有文件
    '''
    files = [os.path.join(root, file) for file in os.listdir(root)]
    files_true = []
    for file in files:
        if not os.path.isfile(file):
            files_true.append(file)
    return files_true


def get_strfile(file_str,pos=-1):
    '''
    得到file_str / or \\ 的最后一个名称
    '''
    endstr_f_filestr = file_str.split('\\')[pos] if '\\' in file_str else file_str.split('/')[pos]
    return endstr_f_filestr

def get_cat_txt(code_txt_path):
    code_name = []
    with open(code_txt_path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if '\n' in line:
                code_name.append(line[:-1])
            else:
                code_name.append(line)
    code_name=sorted(code_name)
    return code_name

def write_txt(text_lst,out_txt=None):
    '''
    每行内容为列表，将其写入text中
    '''
    out_dir =out_txt if out_txt is not None else 'classes.txt'
    file_write_obj = open(out_dir, 'w')  # 以写的方式打开文件，如果文件不存在，就会自动创建
    for text in text_lst:
        file_write_obj.writelines(text)
        file_write_obj.write('\n')
    file_write_obj.close()

    return out_dir
def write_img(img,img_name,out_dir):
    assert img_name[-4:] in img_format,'suffix must be {}'.format(img_format)

    out_dir=build_dir(out_dir)
    cv2.imwrite(os.path.join(out_dir,img_name),img)





def root2lst(read_path):
    '''
    将字符串路径拆成列表的字符串保存
    :param read_path:
    :return:
    '''
    root_slice = []
    for read_path in read_path.split('\\'):
        root_slice.append(read_path)
    root_lst = []
    for slice1 in root_slice:
        for slice in slice1.split('/'):
            root_lst.append(slice)
    return root_lst


def lst2root(root_lst, exept_pos=None):
    if exept_pos is None:
        root_lst = root_lst
    elif exept_pos == -1:
        assert len(root_lst) > 1, 'root_lst count must >=2'
        root_lst = root_lst[:-1]
    else:
        root_lst = root_lst[:exept_pos - 1] + root_lst[exept_pos:]
    result = root_lst[0]
    for root_str in root_lst[1:]:
        result = result + '/' + root_str
    return str(result)

def read_txt(file_path):
    with open(file_path, 'r') as f:
        content = f.read().splitlines()
    return content
def delete_empty_files(read_path):
    '''
    :删除任意路径的空文件
    '''
    count_root = len(root2lst(read_path))
    files_path = build_files(read_path)
    dict_info = {'del_file': [], 'not_del_file': []}
    for filepath in files_path:
        root_file_lst = root2lst(filepath)
        N = count_root - len(root_file_lst)
        if N < 0:
            print('it is problem : ', filepath)
        for i in range(N):
            sub_files_path = lst2root(root_file_lst, exept_pos=-1)
            try:
                os.rmdir(sub_files_path)  # 删除空文件夹
                dict_info['del_file'].append(sub_files_path)
            except:
                dict_info['not_del_file'].append(sub_files_path)
                continue
    print('[del empty info]', dict_info)


def get_pre_dir(root):
    root_new = os.path.abspath(os.path.join(root, ".."))
    return root_new


def get_file_root(root):
    '''
    :return: 寻找root下面有文件夹的路径，输出所有文件夹绝对路径的列表
    '''
    files_lst = [root]
    files_root = []

    if build_files(root) == []:
        files_root = files_lst
    else:
        while True:
            if files_lst != []:
                for path in files_lst:
                    files = build_files(path)
                    for file in files:
                        files_root.append(os.path.join(path, file))
                if build_files(files_root[0]) != []:
                    files_lst = files_root
                    files_root = []
                else:
                    files_lst = []
            else:
                break
    return files_root


def get_files_root(root):
    '''
    :return: 寻找root下面有文件夹的路径，输出所有文件夹绝对路径的列表
    '''
    files_lst = [root]
    result_lst = files_lst
    if build_files(root) == []:
        result_lst = files_lst
    else:
        is_while = True
        files_all_path = [file for file in files_lst]
        while is_while:
            for file_root in files_lst:
                F1 = build_files(file_root)
                for F1 in F1:
                    files_all_path.append(F1)
            is_while = False
            # 排除主文件夹
            record = np.ones((len(files_all_path)))
            for i, F3 in enumerate(files_all_path):
                F3 = files_all_path[i]
                for j, F4 in enumerate(files_all_path):
                    if F3 + '\\' in F4 or F3 + '/' in F4:
                        record[i] = 0
                        break
            # 将需要循环聚集
            files_lst = []
            for i, F3 in enumerate(files_all_path):
                if record[i] == 1:
                    files_lst.append(F3)
            # 判断是否有子文件夹
            for F4 in files_lst:
                file_judge = build_files(F4)
                if file_judge != []:
                    is_while = True
                    break

            result_lst = files_lst

    return result_lst


# excel相关处理

def read_excel(root, sheet_name=1):
    '''
    :param root: excel 路径
    :param sheet_name: 指定excel的sheet
    :return: 通过字典形式返回对应名称的列表
    '''
    df = pd.read_excel(root, sheet_name=sheet_name)
    dict_excel={}
    for key in df.keys():
        dict_excel[key]=df[key].values
    return dict_excel

def read_csv(root):
    '''
    :param root: csv 路径
    :param sheet_name: 指定csv的sheet
    :return: 通过字典形式返回对应名称的列表
    '''
    df = pd.read_csv(root)
    dict_csv={}
    for key in df.keys():
        dict_csv[key]=df[key].values
    return dict_csv



def chinese2img(img, str,  coord=(0, 0),label_size=20,label_color=(255, 0, 0)):
    # 将具有中文的字符打印到图上
    from PIL import Image, ImageDraw, ImageFont
    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
    pilimg = Image.fromarray(cv2img)

    # PIL图片上打印汉字
    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    font = ImageFont.truetype("font/simsun.ttc", label_size, encoding="utf-8")
    # font = ImageFont.truetype("./simhei.ttf", 20, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
    draw.text(tuple(coord), str, label_color, font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
    # PIL图片转cv2 图片
    cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    return cv2charimg


def show_img(img):
    plt.imshow(img)
    plt.show()





def draw_img(img, boxes, labels, scores=None,label_size=20,label_color=(255, 0, 0)):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # 画大矩形
        if scores is not None:
            label = str(labels[i]) + str(scores[i])
        else:
            label = str(labels[i])

        img = chinese2img(img, label, coord=(int(x1 + 5), int(y1 + 4)),label_size=label_size,label_color=label_color)
    return img


def draw_bar_graph(num_dict, out_path):
    '''
    :param num_dict: key为字符串，value为数字
    :param out_path:
    '''

    # 创建一个点数为 8 x 6 的窗口, 并设置分辨率为 80像素/每英寸
    plt.figure(figsize=(28, 20), dpi=100)
    # 再创建一个规格为 1 x 1 的子图
    # plt.subplot(1, 1, 1)
    # 柱子总数
    N = len(num_dict)
    # 包含每个柱子对应值的序列
    values=[]
    code_name=[]
    for code in num_dict.keys():
        code_name.append(code)
        values.append(num_dict[code])
    code_name=tuple(code_name)
    values=tuple(values)

    # values = (56796, 42996, 24872, 13849, 8609, 5331, 1971, 554, 169, 26)
    # 包含每个柱子下标的序列
    index = np.arange(N)
    # 柱子的宽度
    width = 0.8
    # 绘制柱状图, 每根柱子的颜色为紫罗兰色
    p2 = plt.bar(index, values, width, label="num", color="#87CEFA")
    # 设置横轴标签
    plt.xlabel('code')
    # 设置纵轴标签
    plt.ylabel('number of code')
    # 添加标题
    plt.title('Cluster Distribution of codes')
    # 添加纵横轴的刻度
    plt.xticks(index, code_name)

    for a, b in zip(index, values):  ##控制标签位置
        plt.text(a , b + 0.1, '%.1f' % b, ha='center', va='bottom', fontsize=14)


    # plt.xticks(index, (
    # 'mentioned1cluster', 'mentioned2cluster', 'mentioned3cluster', 'mentioned4cluster', 'mentioned5cluster',
    # 'mentioned6cluster', 'mentioned7cluster', 'mentioned8cluster', 'mentioned9cluster', 'mentioned10cluster'))
    # plt.yticks(np.arange(0, 1400, 200))
    # 添加图例
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(out_path,"out_dir.png"))
    plt.show()






def exclude_file(files_root,exclude_file_name):
    # 在filses_root 文件中排除名字未exclude_file_name的文件
    files_root_new=[]
    for file in files_root:

        if exclude_file_name not in  file:
            files_root_new.append(file)
    return files_root_new

def read_json(json_root):
    with open(json_root,'r') as f:
        json_info=json.load(f)

    return json_info






def write_csv(df,out_dir=None):
    if out_dir is not None:
        out_csv=build_dir(os.path.join(out_dir,'out_dir.csv'))
    else:
        out_csv='./out_dir.csv'
    df.to_csv(out_csv)


def show_img(img):
    img=np.array(img,dtype=np.uint8)
    img_shape=img.shape
    if len(img_shape)==2:
        img=np.expand_dims(img,axis=-1).repeat(3,axis=-1)
    plt.imshow(img)
    plt.show()






def find_build_root(root,file_path,out_dir,is_build_dir=True):
    # file_path路径前面字符串被out_dir替换
    out_path=file_path.replace(root,out_dir)
    if is_build_dir:
        out_path=build_dir(out_path)
    return out_path





if __name__ == '__main__':
    root = r'D:\DATA\whtm\data_52902'
    file_path=r'D:\DATA\whtm\data_52902\data1\52902_DRM1.xlsx'
    out_dir=r'D:\DATA'
    out=find_build_root(root, file_path, out_dir, is_build_dir=False)
    print(out)











