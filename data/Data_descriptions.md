## 数据组织说明

为了防止每个epoch开始前对数据集进行加载和预处理，采用硬编码的方式对数据集合进行预处理。

* self.dbImages：包含所有参考图像数据库的图像的相对路径；如['ref/0000000.jpg' 'ref/0000001.jpg' 'ref/0000002.jpg' 'ref/0000003.jpg', 'ref/0000004.jpg' 'ref/0000005.jpg' 'ref/0000006.jpg' 'ref/0000007.jpg', 'ref/0000008.jpg' 'ref/0000009.jpg' 'ref/0000010.jpg' 'ref/0000011.jpg', 'ref/0000012.jpg' 'ref/0000013.jpg']
* self.qImages：包含所有查询数据库图像的相对路径；如['query/0000000.jpg' 'query/0000001.jpg' 'query/0000002.jpg', 'query/0000003.jpg' 'query/0000004.jpg' 'query/0000005.jpg', 'query/0000006.jpg' 'query/0000007.jpg' 'query/0000008.jpg', 'query/0000009.jpg' 'query/0000010.jpg' 'query/0000011.jpg']
* self.qIdx：包含每张查询图像在self.qImages的index值；如 [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17,  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35,  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53,  54  55  56  57  58  59  60  61  62 ]
* self.pIdx：包含每个查询图像对应的多张参考图像在self.dbImages中的index值；如 [array([0, 1]) array([ 9, 10, 11]) array([19, 20, 21]) array([29, 30, 31]), array([39, 40, 41]) array([49, 50, 51]) array([59, 60, 61]), array([69, 70, 71]) array([79, 80, 81]) array([89, 90, 91]), array([ 99, 100, 101]) array([109, 110, 111]) ]
* self.images: 将所有查询和参考图像合并为一个列表，长度为二者之和。self.num_references 记录了参考图像的数量。
* 由self.images构成Dataset
