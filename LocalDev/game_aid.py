
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import copy
import heapq


def cut(image_path, top, bottom, left, right, show=False):
    image = cv2.imread(image_path)
    if image is None:
        print('图像读取失败')
        exit()

    # 裁剪上下无关部分
    cropped_image = image[top:bottom, left:right]

    print('裁剪后的图像大小：', cropped_image.shape)
    return cropped_image



def split(image):
    icon_index = 0
    img_icons = [[0 for _ in range(10)] for _ in range(14)]
    height, width  = image.shape[:2]
    print(f'图像宽度: {width}, 图像高度: {height}')
    icon_width = int(width / 10)
    icon_height = int(height / 14)
    print(f'图标宽度: {icon_width}, 图标高度: {icon_height}')

    for i in range(14):
        for j in range(10):
            x = int(j * icon_width)
            y = int(i * icon_height)
            icon = image[y+15:y+int(icon_height)-15, x+15:x+int(icon_width)-15]
            img_icons[i][j] = icon
            icon_index += 1

    print(f'共保存了{icon_index}个图标')
    return img_icons



def is_background_image(image, edge_threshold=10):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用Canny边缘检测
    edges = cv2.Canny(gray, 100, 200)
    # 计算边缘的数量
    edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
    # 检查边缘数量是否小于阈值
    if edge_density < edge_threshold:
        return False
    return True



def get_valid_icons(img_icons):
    valid_icons = [[0 for _ in range(10)] for _ in range(14)]
    for i in range(14):
        for j in range(10):
            if is_background_image(img_icons[i][j]):
                valid_icons[i][j] = 1
    print(f'共有{np.sum(valid_icons)}个有效图标')
    return valid_icons


def get_loc_fast(board_image, icon_image, show=False):
    template = cv2.cvtColor(icon_image, cv2.COLOR_BGR2GRAY)
    board_gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(board_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(res >= threshold)
    if len(loc[0]) == 0:
        return []
    last_pt = [loc[1][0], loc[0][0]]
    locs = [[last_pt[0], last_pt[1], last_pt[0] + w, last_pt[1] + h]]
    for pt in zip(*loc[::-1]):
        if last_pt[0] + 5 < pt[0] or last_pt[1] + 5 < pt[1]:
            last_pt = pt
            locs.append([pt[0], pt[1], pt[0] + w, pt[1] + h])
    if show:
        for pt in locs:
            cv2.rectangle(board_image, (pt[0], pt[1]), (pt[2], pt[3]), (0, 0, 255), 5)
    return locs
    

def get_board(image, img_icons, valid_icons):
    board = [[0 for _ in range(10)] for _ in range(14)]
    height, width = image.shape[:2]
    icon_width = int(width / 10)
    icon_height = int(height / 14)
    type_index = 1
    valid_icons_cnt = 0
    for r in range(14):
        for c in range(10):
            if not valid_icons[r][c]:
                continue
            if board[r][c] != 0:
                continue

            locs = get_loc_fast(image, img_icons[r][c])
            for loc in locs:
                center_x = int((loc[0] + loc[2]) / 2)
                center_y = int((loc[1] + loc[3]) / 2)
                
                # 直接计算位置
                i = center_x // icon_width
                j = center_y // icon_height
                
                if board[j][i] != 0:
                    continue
                board[j][i] = type_index
                valid_icons_cnt += 1

            type_index += 1
    print(f"识别到{valid_icons_cnt}个图标")
    return board


def get_step(dir, x, y, board):
    step = 0
    nx = x + dir[0]
    ny = y + dir[1]
    while 0 <= nx < 14 and 0 <= ny < 10:
        if board[nx][ny] == 0:
            break
        nx += dir[0]
        ny += dir[1]

    while 0 <= nx < 14 and 0 <= ny < 10 and board[nx][ny] == 0:
        step += 1
        nx += dir[0]
        ny += dir[1]
    return step

def get_pair(x, y, t, board, move_dir):
    pairs = []
    if move_dir[0]:
        dirs = [[0, 1],[0, -1]]
    elif move_dir[1]:
        dirs = [[1, 0],[-1, 0]]
    else:
        dirs = [[0 ,1],[0, -1],[1, 0],[-1, 0]]

    for dir in dirs:
        nx = x + dir[0]
        ny = y + dir[1]
        while 0 <= nx < 14 and 0 <= ny < 10 and board[nx][ny] == 0:
            nx += dir[0]
            ny += dir[1]
        if 0 <= nx < 14 and 0 <= ny < 10 and board[nx][ny] == t:
            pairs.append([nx, ny, dir])
    return pairs

def get_hint(x, y, board):
    ans = []
    t = board[x][y]
    for dir in [[0, 1],[0, -1],[1, 0],[-1, 0]]:
        step = get_step(dir, x, y, board)
        for s in range(1, step + 1):
            nx = x + dir[0] * s
            ny = y + dir[1] * s
            pairs = get_pair(nx, ny, t, board, dir)
            if len(pairs) > 0:
                ans.append([dir, s, pairs])
    pairs = get_pair(x, y, t, board, [0, 0])
    if len(pairs) > 0:
        ans.append([[0, 0], 0, pairs])
    return ans


colors_rgb = [
    (255, 0, 0),     # red
    (0, 0, 255),     # blue
    (0, 255, 0),     # green
    (128, 0, 128),   # purple
    (255, 165, 0),   # orange
    (139, 69, 19),   # brown
    (255, 192, 203), # pink
    (0, 255, 255),   # cyan
    (255, 0, 255),   # magenta
    (255, 255, 0),   # yellow
    (0, 128, 0),     # dark green
    (0, 128, 128),   # teal
    (128, 0, 0),     # maroon
    (128, 128, 0),   # olive
    (75, 0, 130),    # indigo
    (255, 20, 147),  # deep pink
    (0, 0, 128),     # navy
    (245, 245, 220), # beige
    (255, 69, 0),    # orange red
    (154, 205, 50),  # yellow green
    (255, 105, 180), # hot pink
    (47, 79, 79),    # dark slate gray
    (72, 61, 139),   # dark slate blue
    (0, 191, 255),   # deep sky blue
    (105, 105, 105), # dim gray
    (240, 230, 140), # khaki
    (173, 216, 230), # light blue
    (0, 250, 154),   # medium spring green
    (199, 21, 133)   # medium violet red
]



from collections import defaultdict

def get_draw_loc(i, j, bias, dir, w, h):
    # 计算文本位置
    if dir == [0, 1]:  # right
        text_x = (j + 1) * w + bias * 40
        text_y = (i + 0.5) * h + 18
    elif dir == [0, -1]:  # left
        text_x = j * w - 45 - bias * 40
        text_y = (i + 0.5) * h + 18
    elif dir == [1, 0]:  # down
        text_x = (j + 0.5) * w - 18
        text_y = (i + 1) * h + 50 + bias * 60
    elif dir == [-1, 0]:  # up
        text_x = (j + 0.5) * w - 18
        text_y = i * h - 5 - bias * 60
    elif dir == [0, 0]:
        text_x = (j + 0.5) * w - 18
        text_y = (i + 0.5) * h + 18
    return text_x, text_y

def gen_hint_image(image, board):
    height, width = image.shape[:2]
    icon_width = int(width / 10)
    icon_height = int(height / 14)
    # 生成候选颜色，保证每一个图标的框颜色都不一样
    color_used = {}
    color_idx = 0

    for i in range(14):
        for j in range(10):
            if board[i][j] == 0:
                continue
            ops = get_hint(i, j, board)
            if len(ops) > 0:
                if board[i][j] not in color_used:
                    color_used[board[i][j]] = color_idx
                    color_idx += 1
                cdx = color_used[board[i][j]] % len(colors_rgb)
                cv2.rectangle(image, (j * icon_width, i * icon_height), ((j + 1) * icon_width, (i + 1) * icon_height), colors_rgb[cdx], 10)
                step_hint = defaultdict(int)
                for op in ops:
                    dir = op[0]  # [0, 1], [0, -1], [1, 0], [-1, 0] 
                    text_x, text_y = get_draw_loc(i, j, step_hint[tuple(dir)], dir, icon_width, icon_height)
                    step_hint[tuple(dir)] += 1

                    # 绘制文本
                    cv2.putText(image, str(op[1]), (int(text_x), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 2, colors_rgb[cdx], 4, cv2.LINE_AA)
                    
                    for p in op[2]:
                        cv2.rectangle(image, (p[1] * icon_width, p[0] * icon_height), ((p[1] + 1) * icon_width, (p[0] + 1) * icon_height),  colors_rgb[cdx], 10)
    return image

def gen_hint(board):
    hints = []

    for i in range(14):
        for j in range(10):
            if board[i][j] == 0:
                continue
            ops = get_hint(i, j, board)
            for op in ops:
                for p in op[2]:
                    hints.append([[i, j], op[0], op[1], [p[0],p[1]], p[2]])     # [x,y], move_dir, move_step, pair_loc, pair_dir
    return hints

def move(board, x, y, dir, step, pair_loc):
    new_board = copy.deepcopy(board)
    if step == 0:
        new_board[pair_loc[0]][pair_loc[1]] = 0
        new_board[x][y] = 0
        return new_board
    
    s = 1
    while(new_board[x + dir[0] * s][y + dir[1] * s] != 0):
        s += 1
    s -= 1
    for i in range(s, 0, -1):
        new_x = x + dir[0] * (i + step)
        new_y = y + dir[1] * (i + step)
        old_x = x + dir[0] * i
        old_y = y + dir[1] * i
        new_board[new_x][new_y] = new_board[old_x][old_y]
    
    for i in range(step + 1):                   # +1 for pair_A
        new_board[x + dir[0] * i][y + dir[1] * i] = 0
    new_board[pair_loc[0]][pair_loc[1]] = 0     # pair_B
        
    return new_board

def board_to_str(board):
    s = ""
    for i in range(14):
        for j in range(10):
            s += str(board[i][j])
    return s

def cal_score(hints, moved_step):
    active = [[0 for _ in range(10)] for _ in range(14)]
    for h in hints:
        active[h[0][0]][h[0][1]] = 1
    score = (np.sum(active) + moved_step * 2) / (140 - moved_step * 2)
    return score

def main(file_name):
    image = cut(file_name, 500, -530, 33, -33, show=False)
    img_icons = split(image)
    valid_icons = get_valid_icons(img_icons)
    board = get_board(image, img_icons, valid_icons)
    image = gen_hint_image(image, board)
    cv2.imwrite('result/res.jpg', image)


def check(board):
    s = 0
    for i in range(14):
        for j in range(10):
            if board[i][j] != 0:
                s += 1
    return s

def Astar(image):
    img_icons = split(image)
    valid_icons = get_valid_icons(img_icons)
    board = get_board(image, img_icons, valid_icons)
    hints = gen_hint(board)              # [x,y], move_dir, move_step, [pair_x, pair_y], pair_dir
    score = cal_score(hints, 0)
    steps = int(70 - (check(board) / 2))
    q = [[-score, board, hints, steps, []]]             # [-score, board, avi_ops, step, history_ops]
    me = []
    heapq.heapify(q)
    search_cnt = 0
    while len(q) > 0:
        node = heapq.heappop(q)
        search_cnt += 1
        print(f"Step: {node[3]}\tScore: {-node[0]:.2f}\tCheck: {check(node[1])}\tSearch: {search_cnt}")
        for h in node[2]:           # [x,y], move_dir, move_step, [pair_x, pair_y], pair_dir
            new_board = move(node[1], h[0][0], h[0][1], h[1], h[2], h[3])
            if np.sum(new_board) == 0:
                print("Find solution in step: ", node[3] + 1)
                return node[4] + [h]
            if board_to_str(new_board) in me:
                continue
            me.append(board_to_str(new_board))
            
            new_hints = gen_hint(new_board)
            if len(new_hints) == 0:
                continue
            new_score = cal_score(new_hints, node[3] + 1)
            heapq.heappush(q, [-new_score, new_board, new_hints, node[3] + 1, node[4] + [h]])


def get_draw_loc_2(i, j, bias, dir, w, h):
    # 计算文本位置
    if dir == [0, 1]:  # right
        text_x = (j + 1) * w + bias * 40 + 10
        text_y = (i + 0.5) * h + 18
    elif dir == [0, -1]:  # left
        text_x = j * w - 30 - bias * 40
        text_y = (i + 0.5) * h + 14
    elif dir == [1, 0]:  # down
        text_x = (j + 0.5) * w - 13
        text_y = (i + 1) * h + 40 + bias * 60
    elif dir == [-1, 0]:  # up
        text_x = (j + 0.5) * w - 15
        text_y = i * h - 15 - bias * 60
    elif dir == [0, 0]:
        text_x = (j + 0.5) * w - 18
        text_y = (i + 0.5) * h + 50
    return text_x, text_y

def get_solution(file_name):
    image = cut(file_name, 500, -530, 33, -33, show=False)
    op_seq = Astar(image)
    # draw result
    height, width = image.shape[:2]
    w = int(width / 10)
    h = int(height / 14)
    res_image = copy.deepcopy(image)
    for step, op in enumerate(op_seq):
        color = colors_rgb[step % 10]
        # main
        cv2.rectangle(res_image, (op[0][1] * w, op[0][0] * h), ((op[0][1] + 1) * w, (op[0][0] + 1) * h), color, 10)
        cv2.putText(res_image, str(step + 1), (op[0][1] * w + 20, op[0][0] * h + 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 20, cv2.LINE_AA)
        cv2.putText(res_image, str(step + 1), (op[0][1] * w + 20, op[0][0] * h + 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 15, cv2.LINE_AA)
        cv2.putText(res_image, str(step + 1), (op[0][1] * w + 20, op[0][0] * h + 80), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5, cv2.LINE_AA)

        # pair
        cv2.rectangle(res_image, (op[3][1] * w, op[3][0] * h), ((op[3][1] + 1) * w, (op[3][0] + 1) * h), color, 10)
        cv2.putText(res_image, str(step + 1), (op[3][1] * w + 20, op[3][0] * h + 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 20, cv2.LINE_AA)
        cv2.putText(res_image, str(step + 1), (op[3][1] * w + 20, op[3][0] * h + 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 15, cv2.LINE_AA)
        cv2.putText(res_image, str(step + 1), (op[3][1] * w + 20, op[3][0] * h + 80), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5, cv2.LINE_AA)

        # move_step
        text_x, text_y = get_draw_loc_2(op[0][0], op[0][1], 0, op[1], w, h)
        cv2.putText(res_image, str(op[2]), (int(text_x), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 15, cv2.LINE_AA)
        cv2.putText(res_image, str(op[2]), (int(text_x), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 10, cv2.LINE_AA)
        cv2.putText(res_image, str(op[2]), (int(text_x), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 5, cv2.LINE_AA)

        if (step + 1) % 10 == 0:
            cv2.imwrite(f'result/res{int(step/10)}.jpg', res_image)
            print(f"Save result{int(step/10)}.jpg")
            res_image = copy.deepcopy(image)

# main("WechatIMG27.jpg")
get_solution("1651723627137_.pic_hd.jpg")


