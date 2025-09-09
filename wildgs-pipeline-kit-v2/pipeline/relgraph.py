# pipeline/relgraph.py
import numpy as np
import cv2

def connected_components(mask, min_area=50):
    mask_u8 = (mask > 0.5).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    comps = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            comps.append({
                "label_id": i,
                "area": int(area),
                "bbox": stats[i, :4].tolist(),  # x, y, w, h
                "centroid": centroids[i].tolist()
            })
    return comps

def relation_on_under_next_to(table_mask, obj_mask, on_thresh=0.02, near_thresh=0.05):
    """
    纯 2D 关系启发：用 y 方向（图像坐标）近似重力方向
    - on: 目标组件的下边缘接触或紧贴桌面上沿；且组件主体在桌面上方
    - under: 组件主体在桌面下方，且与桌面竖直投影有重叠
    - next_to: 与桌面水平相邻，重叠少但中心距较近
    返回：{'on': [components], 'under': [...], 'next_to': [...]}
    """
    H, W = table_mask.shape
    eps_h = int(H * on_thresh)
    eps_d = int(H * near_thresh)

    tbl = connected_components(table_mask, min_area=500)
    obj = connected_components(obj_mask, min_area=50)
    rel = {"on": [], "under": [], "next_to": []}

    if not tbl or not obj:
        return rel

    # 取面积最大的桌面
    table = max(tbl, key=lambda c: c["area"])
    tx, ty, tw, th = table["bbox"]
    table_top = ty
    table_bottom = ty + th
    table_x1, table_x2 = tx, tx + tw

    for c in obj:
        x, y, w, h = c["bbox"]
        cx, cy = c["centroid"]
        obj_bottom = y + h
        obj_top = y

        # x区间重叠
        overlap_x = not (x + w < table_x1 or x > table_x2)
        # 水平距离（中心）
        horiz_dist = 0.0
        if cx < table_x1:
            horiz_dist = table_x1 - cx
        elif cx > table_x2:
            horiz_dist = cx - table_x2

        on_cond = overlap_x and (abs(obj_bottom - table_top) <= eps_h) and (obj_top <= table_top)
        under_cond = overlap_x and (obj_top >= table_bottom - eps_h)
        next_to_cond = (not overlap_x) and (horiz_dist <= eps_d)

        if on_cond: rel["on"].append(c)
        elif under_cond: rel["under"].append(c)
        elif next_to_cond: rel["next_to"].append(c)

    return rel
