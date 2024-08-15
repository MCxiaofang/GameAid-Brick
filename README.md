# GameAid-Brick
## 功能介绍
本程序为微信小游戏“砖了个砖”的辅助工具，基于Python(cv2)，目前提供两个功能
- 展示当前局面可以消除砖块的操作（通过暴力枚举）
- 展示当前局面可以消除所有砖块的操作步骤（通过A*算法）

## 使用方法
1. 用户使用手机截取当前游戏画面
2. 将截图上传至本程序
3. 程序进行分析和计算
4. 程序返回标注好的图片给用户

## 算法步骤
1. 读取用户上传的图片，截去上下与砖块无关的部分
2. 根据14行10列的标准，分割出140个图标图像
3. 使用cv2的Canny边缘检测，识别其中的背景图标（即没有砖块的空白位置）
4. 使用cv2的matchTemplate模版匹配，识别每个图标在局面中的行列位置
5. 3和4步骤可以得到14*10的二维矩阵，标识了每一个位置的图标类型
6. 以二维矩阵为基础，进行可行操作搜索或解路径搜索

### 可行操作搜索 
1. 通过枚举四个方向上可占用的空白背景的数量，计算每一个砖块的可移动步数
2. 尝试进行移动，并检查移动后是否有匹配消除的对应砖块，记录这些步骤
3. 对所有砖块执行1和2步骤，记录可行操作，绘制在图片上，返回给用户

### 解路径搜索
1. 记录初始局面，及该局面可行的所有操作
2. 创建A*算法的open和close列表，open列表中存放待搜索的局面，close列表中存放已经搜索过的局面
3. 创建局面评估函数f，计算当前局面得分
3. 从open列表中取出评估分最高的局面，进行全部可行操作搜索，生成新的局面，将新局面加入open列表
4. 重复3步骤，直到找到一个局面，所有砖块都被消除

## 实际展示

