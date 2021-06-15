from typing import Tuple
from GA.main_ga import RoomMap, RoomCell, id_from_onehot, H, W

Value = int
Weight = int


def is_types(room_cell: RoomCell, types) -> bool:
    return id_from_onehot(room_cell) in types


def fitness1(room_map: RoomMap) -> Tuple[int, int, int, float]:
    value = 0
    weight = 0
    for h in range(H):
        for w in range(W):
            _value, _weight = heuristic(room_map, h, w)
            value += _value
            weight += _weight
    return value, weight, abs(value - weight), value/weight


def heuristic(room_map: RoomMap, h: int, w: int) -> Tuple[Value, Weight]:
    from GA.main_ga import W, H, one_hot_mapitem
    # TODO: Remove heuristic about overlapping objects.
    value = 0
    weight = 0
    hot_index = -1
    for i, hot in enumerate(room_map[h][w]):
        if hot == 1:
            hot_index = i
            break

    if hot_index == -1:
        return value, weight
    elif hot_index == 0:  # Whiteboard = 2x2,
        weight = 400
        if h == 0 and w <= W-2:
            value += 400
        elif h <= H//3 and w < W-2:
            value += 200
    elif hot_index == 1:  # Projector Screen
        if h == 0 and w <= W - 4:
            value += 600
        elif h <= H // 3 and w <= W - 4:
            value += 300
    elif hot_index == 2:  # Chippendale Table (2x3)
        if h == 0 or h > H-3 or w > W-3:
            room_map[h][w] = one_hot_mapitem(0)
        else:
            weight = 600
            if 1 <= h <= H-4 and 0 <= w <= W-3:
                value += 600
                for _w in range(w, w + 3):
                    if h+2 <= H-1 and is_types(room_map[h + 2][_w], [6,8,14]):
                        value += 600
                    if H-1 >= 0 and is_types(room_map[h - 1][_w], [6,8,14]):
                        value += 600
                for _h in range(h, h + 1):
                    if w+3 <= W-1 and is_types(room_map[_h][w + 3], [6,8,14]):
                        value += 600
                    if w-1 >= 0 and is_types(room_map[_h][w - 1], [6,8,14]):
                        value += 600
                for _h in range(2):
                    for _w in range(3):
                        if is_types(room_map[h+_h][w+_w], [-1, 11,12]):
                            value += 450
                        else:
                            room_map[h + _h][w + _w] = one_hot_mapitem(0)
    elif hot_index == 3:  # TV (Flatscreen)
        weight = 200
        if h == 0 and w < W - 1:
            value = 200
        elif h <= H // 2 and w < W - 1:
            value = 100
    elif hot_index == 4:  # Bookshelf (2x4)
        weight = 800
        if h == 1 and 0 <= w <= W-4:
            value += 800
    elif hot_index == 5 or hot_index == 13:  # Potted Plant (Spikey), Lucky Bamboo
        if h == 0 or h == H-1:
            room_map[h][w] = one_hot_mapitem(0)
        else:
            weight = 200
            if h == 1 or h == H-2:
                value += 200
            if w == 0 or w == W-1:
                value += 200
    elif hot_index == 6 or hot_index == 14:  # Mod Chair, Dining Chair (Square)
        if h == 0 or h == H-1:
            room_map[h][w] = one_hot_mapitem(0)
        else:
            weight = 100
            if 1 <= h <= H-2:
                value = 300
    elif hot_index == 7:  # Captain’s Chair
        if h == 0 or h == H-1:
            room_map[h][w] = one_hot_mapitem(0)
        else:
            weight = 300
            if 1 <= h <= H//3:
                value += 300
    elif hot_index == 8:  # Chair (Simple)
        if h == 0 or h == H-1:
            room_map[h][w] = one_hot_mapitem(0)
        else:
            weight = 100
            if 2 <= h <= H-2 and 1 <= w <= W-2:
                value += 200
                for _w in range(1, W-2):
                    if _w != w:
                        value += 1000
                        break
    elif hot_index == 9:  # Chippendale Table (3x3)
        if h == 0 or h > H-4 or w > W-3:
            room_map[h][w] = one_hot_mapitem(0)
        else:
            weight = 900
            if 1 <= h <= H-4 and w <= W-3:
                value = 900
                for _w in range(w, w + 3):
                    if h+3 <= H-1 and is_types(room_map[h + 3][_w], [6,8,14]):
                        value += 900
                    if h-1 >= 0 and is_types(room_map[h - 1][_w], [6,8,14]):
                        value += 900
                for _h in range(h, h + 2):
                    if w+3 <= W-1 and is_types(room_map[_h][w + 3], [6,8,14]):
                        value += 900
                    if w-1 >= 0 and is_types(room_map[_h][w - 1], [6,8,14]):
                        value += 900
                for _h in range(3):
                    for _w in range(3):
                        if is_types(room_map[h+_h][w+_w], [-1, 11,12]):
                            value += 450
                        else:
                            room_map[h + _h][w + _w] = one_hot_mapitem(0)
    elif hot_index == 10:  # Bookshelf [Tall] (1x2)
        if h == 0 or h == H-1:
            room_map[h][w] = one_hot_mapitem(0)
        else:
            weight = 200
            if h == 1:
                value += 200
    elif hot_index == 11:  # Laptop
        pass
    elif hot_index == 12:  # Microphone
        pass
    return value, weight
