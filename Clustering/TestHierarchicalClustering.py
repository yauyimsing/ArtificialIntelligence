# -*- coding:utf-8 -*-

from PIL import ImageDraw, Image
import numpy as np
import os
import sys

nodeList = []  # 用于存储所有的节点，包含图片节点，与聚类后的节点
distance = {}  # 用于存储所有每两个节点的距离，数据格式{(node1.id,node2.id):30.0,(node2.id,node3.id):40.0}


class node:
    def __init__(self, data):
        '''每个样本及样本合并后节点的类
            data：接受两种格式，
            1、当为字符（string）时，是图片的地址，同时也表示这个节点就是图片
            2、合并后的类，传入的格式为(leftNode,rightNode) 即当前类表示合并后的新类，而对应的左右节点就是子节点
        '''
        self.id = len(nodeList)  # 设置一个ID,以nodeList当然长度为ID,在本例中ID本身没太大用处，只是如果看代码时，有时要看指向时有点用
        self.parent = None  # 指向合并后的类
        self.pos = None  # 用于最后绘制节构图使用，赋值时为(x,y,w,h)格式
        if type(data) == type(""):
            '''节点为图片'''
            self.imgData = Image.open(data)
            self.left = None
            self.right = None
            self.level = 0  # 图片为最终的子节点，所有图片的层级都为0，设置层级是为了最终绘制结构图

            npTmp = np.array(self.imgData).reshape(-1, 3)  # 将图片数据转化为numpy数据，shape为(高，宽，3)，3为颜色通道
            npTmp = npTmp.reshape(-1, 3)  # 重新排列，shape为(高*宽，3)
            self.feature = npTmp.mean(axis=0)  # 计算RGB三个颜色通道均值

        else:
            '''节点为合成的新类'''
            self.imgData = None
            self.left = data[0]
            self.right = data[1]
            self.left.parent = self
            self.right.parent = self

            self.level = max(self.left.level, self.right.level) + 1  # 层级为左右节高层级的级数+1
            self.feature = (self.left.feature + self.right.feature) / 2  # 两类的合成一类时，就是左右节点的feature相加/2

        # 计算该类与每个其他类的距离，并存入distance
        for x in nodeList:
            distance[(x, self)] = np.sqrt(np.sum((x.feature - self.feature) ** 2))

        nodeList.append(self)  # 将本类加入nodeList变量

    def drawNode(self, img, draw, vLineLenght):
        # 绘制结构图
        if self.pos == None: return
        if self.left == None:
            # 如果是图片
            self.imgData.thumbnail((self.pos[2], self.pos[3]))
            img.paste(self.imgData, (self.pos[0], self.pos[1]))
            draw.line((int(self.pos[0] + self.pos[2] / 2)
                       , self.pos[1] - vLineLenght
                       , int(self.pos[0] + self.pos[2] / 2)
                       , self.pos[1])
                      , fill=(255, 0, 0))
        else:
            # 如果不是图片
            draw.line((int(self.pos[0])
                       , self.pos[1]
                       , int(self.pos[0] + self.pos[2])
                       , self.pos[1])
                      , fill=(255, 0, 0))

            draw.line((int(self.pos[0] + self.pos[2] / 2)
                       , self.pos[1]
                       , int(self.pos[0] + self.pos[2] / 2)
                       , self.pos[1] - self.pos[3])
                      , fill=(255, 0, 0))


def loadImg(path):
    '''path 图片目录，根据自己存的地方改写'''
    files = None
    try:
        files = os.listdir(path)
    except:
        print('未正确读取目录：' + path + ',图片目录，请根据自己存的地方改写,并保证没有hierarchicalResult.jpg,该文件为最后生成文件')
        return None
    for i in files:

        if os.path.splitext(i)[1].lower() == '.jpg' and os.path.splitext(i)[0].lower() != 'hierarchicalresult':
            fileName = os.path.join(path, i)
            node(fileName)
            print(node)
    return os.path.join(path, 'hierarchicalResult.jpg')


def getMinDistance():
    '''从distance中过滤出未分类的结点，并读取最小的距离'''
    vars = list(filter(lambda x: x[0].parent == None and x[1].parent == None, distance))
    minDist = vars[0]
    for x in vars:
        if minDist == None or distance[x] < distance[minDist]:
            minDist = x
    return minDist


def createTree():
    while len(list(filter(lambda x: x.parent == None, nodeList))) > 1:  # 合并到最后时，只有一个类，只要有两个以上未合并，就循环
        minDist = getMinDistance()
        # 创建非图片的节点，之所以把[1]做为左节点，因为绘图时的需要，
        # 在不断的产生非图片节点时，在nodeList的后面的一般是新节点，但绘图时绘在左边
        node((minDist[1], minDist[0]))
    return nodeList[-1]  # 最后一个插入的节点就是要节点


def run():
    root = createTree()  # 创建树结构

    # 一句话的PYTON，实现二叉树的左右根遍历，通过通过遍历，进行排序后，取出图片，做为最底层的打印
    sortTree = lambda node: ([] if node.left == None else sortTree(node.left)) + (
    [] if node.right == None else sortTree(node.right)) + [node]
    treeTmp = sortTree(root)
    treeTmp = list(filter(lambda x: x.left == None, treeTmp))  # 没有左节点的，即为图片

    thumbSize = 60  # 缩略图的大小，，在60X60的小格内缩放
    thumbSpace = 20  # 缩略图间距
    vLineLenght = 80  # 上下节点，即每个level之间的高度

    imgWidth = len(treeTmp) * (thumbSize + thumbSpace)
    imgHeight = (root.level + 1) * vLineLenght + thumbSize + thumbSpace * 2
    img = Image.new('RGB', (imgWidth, imgHeight), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    for item in enumerate(treeTmp):
        # 为所有图片增加绘图数据
        x = item[0] * (thumbSize + thumbSpace) + thumbSpace / 2
        y = imgHeight - thumbSize - thumbSpace / 2 - ((item[1].parent.level - 1) * vLineLenght)
        w = item[1].imgData.width
        h = item[1].imgData.height
        if w > h:
            h = h / w * thumbSize
            w = thumbSize
        else:
            w = w / h * thumbSize
            h = thumbSize
            x += (thumbSize - w) / 2
        item[1].pos = (int(x), int(y), int(w), int(h))
        item[1].drawNode(img, draw, vLineLenght)

    for x in range(1, root.level + 1):
        # 为所有非图片增加绘图的数据
        items = list(filter(lambda i: i.level == x, nodeList))
        for item in items:
            x = item.left.pos[0] + (item.left.pos[2] / 2)
            w = item.right.pos[0] + (item.right.pos[2] / 2) - x
            y = item.left.pos[1] - (item.level - item.left.level) * vLineLenght
            h = ((item.parent.level if item.parent != None else item.level + 1) - item.level) * vLineLenght
            item.pos = (int(x), int(y), int(w), int(h))
            item.drawNode(img, draw, vLineLenght)
    img.save(resultFile)


resultFile = loadImg(r".\picture")  # 读取数据，并返回最后结果要存储的文件名，目录根据自己存的位置进行修改
if resultFile != 'None':
    run()
    print("结构图生成成功，最终结构图存储于：" + resultFile)
