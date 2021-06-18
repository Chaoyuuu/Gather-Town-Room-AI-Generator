import os
from datetime import datetime
from functools import partial
from random import choice, choices, random, randint
from typing import List, Tuple, ClassVar
from object_dictionary import object_abbrev_name_dict
from fitness.gan import Discriminator
import json

# Defining Type
RoomCell = List[int]  # (0~14)=one-hot; (15,16)=direction.
RoomMap = List[List[RoomCell]]
Population = List[RoomMap]

# Constants (Common)
W = 10
H = 13
P = 0.0
M_P1 = 0.5
M_P2 = 0.2
SURVIVE_PAIR = 1
POP_SIZE = 30
GEN_LIMIT = 30000
WORK_SIZE = 30
MIN_ITEMS = 10
MAX_ITEMS = 20
# Constants (fitness1)
WEIGHT_LIMIT = 20000
# Constants (fitness2)
SOME = 5
GAN_BASELINE = 0.4
DISS = Discriminator()

# Mappings
orientations = [
    [0, 1],  # N
    [0, -1],  # S
    [1, 0],  # E
    [-1, 0],  # W
]
itemname_from_id = {
    -1: "地板",
    0: "Whiteboard",
    1: "Projector Screen",
    2: "Chippendale Table (2x3)",
    3: "TV (Flatscreen)",
    4: "Bookshelf (2x4)",
    5: "Potted Plant (Spikey)",
    6: "Mod Chair",
    7: "Captain's Chair",
    8: "Chair (Simple)",
    9: "Chippendale Table (3x3)",
    10: "Bookshelf [Tall] (1x2)",
    11: "Laptop",
    12: "Microphone",
    13: "Lucky Bamboo",
    14: "Dining Chair (Square)"
}
id_from_itemname = {
    "地板": -1,
    "Whiteboard": 0,
    "Projector Screen": 1,
    "Chippendale Table (2x3)": 2,
    "TV (Flatscreen)": 3,
    "Bookshelf (2x4)": 4,
    "Potted Plant (Spikey)": 5,
    "Mod Chair": 6,
    "Captain's Chair": 7,
    "Chair (Simple)": 8,
    "Chippendale Table (3x3)": 9,
    "Bookshelf [Tall] (1x2)": 10,
    "Laptop": 11,
    "Microphone": 12,
    "Lucky Bamboo": 13,
    "Dining Chair (Square)": 14
}


# Preparing & Helper functions
def count_all_furniture() -> int:
    return len(itemname_from_id)-1  # Excluding the floor.


def id_from_onehot(one_hot: RoomCell) -> int:
    for i in range(count_all_furniture()):
        if one_hot[i]:
            return i
    return -1


def one_hot_mapitem(appear_prob: float) -> RoomCell:
    arr = [0] * count_all_furniture()
    if random() >= appear_prob:  # random() in [0, 1)
        arr.extend([0, 0])
        return arr
    arr[randint(0, count_all_furniture()-1)] = 1
    arr.extend(choice(orientations))
    return arr


def one_hot_by_id(id: int) -> RoomCell:
    arr = [0] * count_all_furniture()
    if id == -1:
        arr.extend([0, 0])
    else:
        arr[id] = 1
        arr.extend(choice(orientations))
    return arr


def room_map_empty(h: int, w: int) -> RoomMap:
    return [[one_hot_mapitem(0) for _ in range(w)] for _ in range(h)]


def room_map_random(h: int, w: int, appear_prob: float) -> RoomMap:
    return [[one_hot_mapitem(appear_prob) for _ in range(w)] for _ in range(h)]


def random_population(size: int, h: int, w: int, appear_prob: float) -> Population:
    return [room_map_random(h, w, appear_prob) for _ in range(size)]


def pretty_print(room_map: RoomMap) -> None:
    for h in range(H):
        arr = []
        for w in range(W):
            arr.append(object_abbrev_name_dict[id_from_onehot(room_map[h][w])])
        print(arr)


# GA functions
def construct_DISS() -> ClassVar[Discriminator]:
    import torch
    print("Initializing GAN discriminator...")
    DISS.load_state_dict(torch.load("../GAN/saveD.pt"))
    DISS.eval()
    return DISS


def fitness_GAN_DISS(room_map: RoomMap, DISS: ClassVar[Discriminator]) -> float:
    import numpy as np
    import torch
    # Transpose dimension from (13, 10, 17) to (17, 13, 10).
    if count_room_map_items(room_map) == 0:
        return 0.2
    np_room_map = np.array(room_map, dtype="float32").transpose((2, 0, 1))
    torch_room_map = torch.from_numpy(np_room_map.reshape([1, *np_room_map.shape]))
    # pick [0] from the result array, which is of [0]-th input data.
    return DISS.forward(torch_room_map)[0][0]


def count_room_map_items(room_map: RoomMap) -> int:
    from GA.fitness.heu import is_types
    count = 0
    for h in range(H):
        for w in range(W):
            if not is_types(room_map[h][w], [-1]):
                count += 1
    return count


def selection_by_fitness(population: Population, fitness_func) -> Population:
    return choices(
        population=population,
        weights=[fitness_func(room_map) for room_map in population],
        k=2
    )


def crossover_xy_divide_2(a: RoomMap, b: RoomMap) -> Tuple[RoomMap, RoomMap]:
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise ValueError("RoomMap should have the same (h, w)")

    for h in range(H//2):
        for w in range(W):
            a[h][w], b[h][w] = b[h][w], a[h][w]

    for h in range(H):
        for w in range(W//2):
            a[h][w], b[h][w] = b[h][w], a[h][w]

    return a, b


def crossover_y_divide_4(a: RoomMap, b: RoomMap) -> Tuple[RoomMap, RoomMap]:
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise ValueError("RoomMap should have the same (h, w)")
    # 橫切四互換, 內外換: 換(2,4)
    for h in range(H//4, H//2):
        for w in range(W):
            a[h][w], b[h][w] = b[h][w], a[h][w]

    for h in range(3*H//4, H):
        for w in range(W):
            a[h][w], b[h][w] = b[h][w], a[h][w]

    return a, b


def mutation_random_add(room_map: RoomMap, prob: float, remove: bool = False) -> None:
    if random() >= prob:
        return
    rh = randint(0, H - 1)
    rw = randint(0, W - 1)
    if remove:
        room_map[rh][rw] = one_hot_by_id(-1)
    else:
        room_map[rh][rw] = one_hot_mapitem(appear_prob=1)


def mutation_y_shift(room_map: RoomMap, prob: float) -> None:
    from fitness.heu import is_types
    if random() >= prob:
        return

    # y-shift - division 1 move down
    for h in range(H//2-1, H//4-1, -1):
        for w in range(W):
            if not is_types(room_map[h][w], [-1]):
                room_map[h+1][w] = room_map[h][w]
                room_map[h][w] = one_hot_by_id(-1)

    # y-shift 2 - division 2 move up
    for h in range(H//2+1, 3*H//4):
        for w in range(W):
            if not is_types(room_map[h][w], [-1]):
                room_map[h-1][w] = room_map[h][w]
                room_map[h][w] = one_hot_by_id(-1)

    # y-shift 3 - randomly move middle up/down
    for w, tile in enumerate(room_map[H//2]):
        h_offset = choice([-1, 1])
        if random() < 0.5 and not is_types(tile, [-1]):
            room_map[H//2+h_offset][w] = room_map[H//2][w]
            room_map[H//2][w] = one_hot_by_id(-1)


def mutation_chairset_swap(room_map: RoomMap, prob: float) -> None:
    from GA.fitness.heu import is_types
    if random() >= prob:
        return
    # 椅子換整組
    for h in range(H):
        for w in range(W):
            if is_types(room_map[h][w], [id_from_itemname["Chair (Simple)"]]):
                room_map[h][w] = one_hot_by_id(id_from_itemname["Dining Chair (Square)"])
            elif is_types(room_map[h][w], [id_from_itemname["Dining Chair (Square)"]]):
                room_map[h][w] = one_hot_by_id(id_from_itemname["Chair (Simple)"])


def run_ga(
        populate_func,
        fitness_func,
        selection_func,
        crossover_func,
        mutation_func1,
        mutation_func2,
        generation_limit: int) -> Tuple[Population, int]:
    # Prepare initial data.
    population = sorted(populate_func(),
                        key=lambda room_map: fitness_func(room_map),
                        reverse=True)

    i = 1
    while i <= generation_limit:
        print(f"GEN={i}, len(population)={len(population)}")

        # if fitness_func(population[0]) > GAN_BASELINE:

        next_generation = population[:2*SURVIVE_PAIR]  # pick 2 elites first

        for j in range(len(population)//2 - SURVIVE_PAIR):
            parents = selection_func(population, fitness_func)
            for child in crossover_func(parents[0], parents[1]):
                mutation_func1(child)
                while True:
                    if count_room_map_items(child) < MIN_ITEMS:
                        mutation_func1(child)
                    elif count_room_map_items(child) > MAX_ITEMS:
                        mutation_func1(child, remove=True)
                    else:
                        break
                mutation_func2(child)
                next_generation.append(child)

        print(f"\tDISS={fitness_func(next_generation[0])}, room={pretty_print(next_generation[0])}")
        if some_pass_by_fitness(SOME, next_generation, fitness_func):
            print(f"Top {SOME} of population has passed GAN baseline={GAN_BASELINE}, finished!")
            break

        # Sort population by GAN DISS's value in descending order.
        population = sorted(next_generation,
                            key=lambda room_map: fitness_func(room_map),
                            reverse=True)
        i+=1

    return population, i-1


def some_pass_by_fitness(some: int, population: Population, fitness_func) -> bool:
    for i in range(some):
        if fitness_func(population[i]) < GAN_BASELINE:
            return False
    return True


if __name__ == '__main__':
    from GA.prepare_data.workspace import workspace_population, dict_from_roommap

    population, gens = run_ga(
        # populate_func=partial(workspace_population, size=WORK_SIZE),
        populate_func=partial(random_population, size=POP_SIZE, h=H, w=W, appear_prob=P),
        fitness_func=partial(fitness_GAN_DISS, DISS=construct_DISS()),
        selection_func=selection_by_fitness,
        crossover_func=crossover_y_divide_4,
        mutation_func1=partial(mutation_random_add, prob=M_P1),
        mutation_func2=partial(mutation_y_shift, prob=M_P2),
        generation_limit=GEN_LIMIT
    )

    print(f"------------generation------------:\n{gens}")
    for i in range(10):
        pretty_print(population[i])
        print("---")
    print(f"------------best population------------:\n")
    pretty_print(population[0])

    # Append to file
    with open('ga_store.txt', 'a') as f_in:
        for room_map in population[:6]:
            for h in range(H):
                arr = []
                for w in range(W):
                    arr.append(object_abbrev_name_dict[id_from_onehot(room_map[h][w])])
                f_in.write(json.dumps(arr))
            f_in.write(f'  DISS={fitness_GAN_DISS(room_map, DISS)}\n')
            f_in.write(f'------\n')
        f_in.write(f'^^^-------{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}-------^^^\n')

    # Append to json
    with open("output.json", "r+") as f:
        json_list = []
        if os.stat("output.json").st_size > 0:
            json_list = json.load(f)
        for room_map in population[:SOME]:
            json_list.append(dict_from_roommap(room_map))
        # print(f"debug---json_list={json_list}")
        f.seek(0)
        json.dump(json_list, f)
        f.truncate()

