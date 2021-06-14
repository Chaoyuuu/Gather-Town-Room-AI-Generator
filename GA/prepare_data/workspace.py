from pprint import pprint
from GA.main_ga import RoomMap, Population
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

    for m in range(TOTAL_MEMBERS):
        for i_json_room_from_m in range(EACH_ONE_DRAWN):
            room_map = from_jsonroom_to_roommap(json_rooms[m*EACH_ONE_DRAWN+i_json_room_from_m]["room"])
            room_maps.append(room_map)
            # pprint(room_map)


def from_jsonroom_to_roommap(json_room) -> RoomMap:
    from GA.main_ga import mapitemname_to_id, room_map_empty, H, W
    room_map = room_map_empty(H, W)
    for item in json_room:
        one_hot = [0] * 15
        one_hot[mapitemname_to_id[item["_name"]]] = 1
        one_hot.extend(orientation_to_tuple[item["orientation"]])
        # print(f"item_name={item['_name']}, one_hot={one_hot}")
        room_map[item["y"]][item["x"]] = one_hot
    return room_map


if __name__ == '__main__':
    prepare()