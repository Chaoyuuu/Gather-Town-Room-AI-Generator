from GA.main_ga import RoomMap, Population, RoomCell
from random import choices
import requests


# Constants
TOTAL_MEMBERS = 3
EACH_ONE_DRAWN = 9*7
orientation_to_tuple = {
    0: [0, -1],
    1: [1, 0],
    2: [0, 1],
    3: [-1, 0]
}
# Data
room_maps = []
ChaoyuTrain = 0
NingTrain = 1
HanTrain = 2


def workspace_population(size: int) -> Population:
    prepare()
    return choices(room_maps, k=size)


def prepare():
    url = "https://raw.githubusercontent.com/Chaoyuuu/Gather-Town-Datasets/master/datasets.json"
    json_rooms = requests.get(url).json()
    global room_maps
    for m in range(TOTAL_MEMBERS):
        for i_json_room_from_m in range(EACH_ONE_DRAWN):
            room_map = from_jsonroom_to_roommap(json_rooms[m*EACH_ONE_DRAWN+i_json_room_from_m]["room"])
            room_maps.append(room_map)
            # pprint(room_map)


def from_jsonroom_to_roommap(json_room) -> RoomMap:
    from GA.main_ga import id_from_itemname, room_map_empty, H, W
    room_map = room_map_empty(H, W)
    for item in json_room:
        one_hot = [0] * 15
        one_hot[id_from_itemname[item["_name"]]] = 1
        one_hot.extend(orientation_to_tuple[item["orientation"]])
        # print(f"item_name={item['_name']}, one_hot={one_hot}")
        room_map[item["y"]][item["x"]] = one_hot
    return room_map


def dict_from_roommap(room_map: RoomMap):
    from GA.main_ga import id_from_onehot, itemname_from_id, H, W
    from object_dictionary import object_dict
    from GA.fitness.heu import is_types
    from utils import decode_orientation
    json_room = {"generator": "ga", "room": []}
    for h in range(H):
        for w in range(W):
            if not is_types(room_map[h][w], [-1]):
                item = {}
                item["x"] = w
                item["y"] = h
                item["_name"] = itemname_from_id[id_from_onehot(room_map[h][w])]
                item["orientation"] = decode_orientation(id_from_onehot(room_map[h][w]), room_map[h][w][-2], room_map[h][w][-1])
                item["width"] = object_dict[item["_name"]][item["orientation"]]["width"]
                item["height"] = object_dict[item["_name"]][item["orientation"]]["height"]
                item["normal"] = object_dict[item["_name"]][item["orientation"]]["normal"]
                json_room["room"].append(item)

    return json_room


def orientation_from_onehot(one_hot: RoomCell):
    x, y = one_hot[-2:]
    if x == -1 and y == 0:
        return 1
    if x == 1 and y == 0:
        return 3
    if x == 0 and y == 1:
        return 2
    if x == 0 and y == -1:
        return 0


if __name__ == '__main__':
    prepare()