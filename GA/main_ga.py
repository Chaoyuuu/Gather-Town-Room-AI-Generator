from functools import partial
from random import choice, choices, random, randint
from typing import List, Tuple, ClassVar
from pprint import pprint
from fitness.gan import Discriminator

# Defining Type
RoomCell = List[int]  # (0~14)=one-hot; (15,16)=direction.
RoomMap = List[List[RoomCell]]
Population = List[RoomMap]

# Constants (Common)
W = 10
H = 13
P = 0.01
M_P1 = 0.01
M_P2 = 0.5
POP_SIZE = 63
GEN_LIMIT = 200
# Constants (fitness1)
WEIGHT_LIMIT = 20000
# Constants (fitness2)
SOME = 2
GAN_BASELINE = 0.9

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
    7: "Captain\"s Chair",
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
        for w in range(W):
            print(itemname_from_id[id_from_onehot(room_map[h][w])], end=" ")
        print("")


# GA functions
def construct_DISS() -> ClassVar[Discriminator]:
    import torch
    print("Initializing GAN discriminator...")
    DISS = Discriminator()
    DISS.load_state_dict(torch.load("./fitness/save.pt"))
    DISS.float()
    DISS.eval()
    return DISS


def fitness_GAN_DISS(room_map: RoomMap, DISS: ClassVar[Discriminator]) -> int:
    import numpy as np
    import torch
    np_room_map = np.array(room_map, dtype="float32").transpose((2, 0, 1))
    torch_room_map = torch.from_numpy(np_room_map.reshape([1, *np_room_map.shape]))
    # pick [0] from the result array, which is of [0]-th input data.
    return DISS.forward(torch_room_map)[0][0]


def selection_by_fitness(population: Population, fitness_func) -> Population:
    return choices(
        population=population,
        weights=[fitness_func(room_map) for room_map in population],
        k=2
    )


def crossover_xy_divide_2(a: RoomMap, b: RoomMap) -> Tuple[RoomMap, RoomMap]:
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise ValueError("RoomMap should have the same (w, h)")
    new_a = a
    new_b = b
    new_a[H // 2:], new_b[H // 2:] = new_b[H // 2:], new_a[H // 2:]
    for i in range(H):
        new_a[i][W // 2:], new_b[i][W // 2:] = new_b[i][W // 2:], new_a[i][W // 2:]
    return new_a, new_b


def crossover_y_divide_4(a: RoomMap, b: RoomMap) -> Tuple[RoomMap, RoomMap]:
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise ValueError("RoomMap should have the same (w, h)")
    from fitness.heu import is_types
    new_a = a
    new_b = b
    # 橫切四互換, 內外換: 留(1,3), 換(2,4)
    new_a[:H//4], new_b[:H//4] = new_b[:H//4], new_a[:H//4]
    new_a[H//2:H//2+H//4], new_b[H//2:H//2+H//4] = new_b[H//2:H//2+H//4], new_a[H//2:H//2+H//4]

    # 椅子換整組
    for h in range(H):
        for w in range(W):
            # for new_a
            if is_types(new_a[h][w], [id_from_itemname["Chair (Simple)"]]):
                new_a[h][w] = one_hot_by_id(id_from_itemname["Dining Chair (Square)"])
            elif is_types(new_a[h][w], [id_from_itemname["Chair (Simple)"]]):
                new_a[h][w] = one_hot_by_id(id_from_itemname["Dining Chair (Square)"])
            # for new_b
            if is_types(new_b[h][w], [id_from_itemname["Chair (Simple)"]]):
                new_b[h][w] = one_hot_by_id(id_from_itemname["Dining Chair (Square)"])
            elif is_types(new_a[h][w], [id_from_itemname["Chair (Simple)"]]):
                new_b[h][w] = one_hot_by_id(id_from_itemname["Dining Chair (Square)"])

    return new_a, new_b


def mutation(room_map: RoomMap, prob: float) -> None:
    from fitness.heu import is_types
    if random() >= prob:
        return
    ri = randint(0, H - 1)
    rj = randint(0, W - 1)
    # TODO: This heuristic might not be needed.
    if is_types(room_map[ri][rj], [-1]):
        room_map[ri][rj] = one_hot_mapitem(1)


def mutation_y_shift(room_map: RoomMap, prob: float) -> None:
    from fitness.heu import is_types
    if random() >= prob:
        return
    # y-shift
    top = H * 3 // 4 - 1
    down = H // 4
    for h in range(H // 4, H * 3 // 4):
        for w in range(W):
            if not is_types(room_map[h][w], [-1]):
                top = min(top, h)
                down = max(down, h)
    dice = randint(0, 1)
    if dice == 0 and top > H // 4:  # Can move top
        for h in range(H // 4, H * 3 // 4 - 1):
            for w in range(W):
                room_map[h][w] = room_map[h + 1][w]
                room_map[h + 1][w] = one_hot_mapitem(0)
    elif dice == 1 and down < H * 3 // 4 - 1:  # Can move down
        for h in range(H // 4 + 1, H * 3 // 4):
            for w in range(W):
                room_map[h][w] = room_map[h - 1][w]
                room_map[h - 1][w] = one_hot_mapitem(0)


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

    i = 0
    for i in range(generation_limit):
        print(f"GEN={i + 1}, len(population)={len(population)}")

        print(f"\tDISS={fitness_func(population[0])}, room={pretty_print(population[0])}")
        if some_pass_by_fitness(SOME, population, fitness_func):
            print(f"Top {SOME} of population has passed GAN baseline={GAN_BASELINE}, finished!")
            break

        next_generation = population[:2]  # pick elites first

        for j in range(len(population)//2 - 1):
            parents = selection_func(population, fitness_func)
            c1, c2 = crossover_func(parents[0], parents[1])
            mutation_func1(c1)
            mutation_func1(c2)
            mutation_func2(c2)
            mutation_func2(c2)
            next_generation.extend([c1, c2])

        # Sort population by GAN DISS's value in descending order.
        population = sorted(next_generation,
                            key=lambda room_map: fitness_func(room_map),
                            reverse=True)

    return population, i


def some_pass_by_fitness(some:int, population: Population, fitness_func) -> bool:
    for i in range(some):
        if fitness_func(population[i]) < GAN_BASELINE:
            return False
    return True


if __name__ == '__main__':

    population, gens = run_ga(
        populate_func=partial(random_population, size=POP_SIZE, h=H, w=W, appear_prob=P),
        fitness_func=partial(fitness_GAN_DISS, DISS=construct_DISS()),
        selection_func=selection_by_fitness,
        crossover_func=crossover_y_divide_4,
        mutation_func1= partial(mutation, prob=M_P1),
        mutation_func2=partial(mutation_y_shift, prob=M_P2),
        generation_limit=GEN_LIMIT
    )

    print(f"------------generation------------:\n{gens}")
    pprint(f"------------best population------------:\n")
    # TODO: 將每次的結果 Population 加到輸出檔案結尾
    pretty_print(population[0])
