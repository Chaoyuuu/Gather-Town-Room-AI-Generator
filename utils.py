def decode_orientation(object_index, x, y):
    down = {'x': 0, 'y': 1}
    up = {'x': 0, 'y': -1}
    left = {'x': -1, 'y': 0}
    right = {'x': 1, 'y': 0}
    orientation = [down, up, left, right]

    two_direction_object = [2]
    one_direction_object = [0, 1, 3, 5, 9, 13]
    four_direction_object = [4, 6, 7, 8, 10, 11, 12, 14]

    if object_index in one_direction_object:
        return 0
    elif object_index in two_direction_object:
        up_length = abs(x - up.get('x')) ** 2 + abs(y - up.get('y')) ** 2
        down_length = abs(x - down.get('x')) ** 2 + abs(y - down.get('y')) ** 2
        if up_length < down_length:
            return 2
        return 0
    elif object_index in four_direction_object:
        min_length = 0
        object_orientation = 0
        for i in range(len(orientation)):
            length = abs(x - orientation[i].get('x')) ** 2 + abs(y - orientation[i].get('y')) ** 2
            if length < min_length:
                min_length = length
                object_orientation = i
        return object_orientation
    else:
        print("error in orientation")
        return 0


def ignore_chair(index):
    if index == 8 or index == 14:
        return True
    return True


def shift_object(index, x, orientation):
    if index == 9 or index == 2:
        x -= 1
    elif index == 7:
        if orientation == 0 or orientation == 2:
            x -= 1
    return x