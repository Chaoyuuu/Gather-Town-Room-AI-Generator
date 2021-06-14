from random import choice, randint, choices, random
from typing import List, Callable, Tuple, Optional
from pprint import pprint

# Defining Type
RoomCell = List[int]  # 0~14=種類one-hot; (15,16)=方向
RoomMap = List[List[RoomCell]]
Population = List[RoomMap]

# Functions
# FitnessFunc = Callable[[RoomMap], int]  # fitness_adapter
# PopulateFunc = Callable[[int, int, int, float], Population]  # random_population
# SelectionFunc = Callable[[Population, FitnessFunc], Population]  # select_parents_pair
# CrossoverFunc = Callable[[RoomMap, RoomMap], Tuple[RoomMap, RoomMap]]  # single_point_crossover
# MutationFunc = Callable[[RoomMap, Optional[float]], None]  # mutation

# Constants & Mapping
W = 10
H = 13
P = 0.4
M_P = 0.05
WEIGHT_LIMIT = 20000
P_SIZE = 63
GEN_LIMIT = 200
orientations = [
    [0, 1],  # N
    [0, -1],  # S
    [1, 0],  # E
    [-1, 0],  # W
]
id_to_mapitemname = {
    -1: '地板',
    0: 'Whiteboard',
    1: 'Projector Screen',
    2: 'Chippendale Table (2x3)',
    3: 'TV (Flatscreen)',
    4: 'Bookshelf (2x4)',
    5: 'Potted Plant (Spikey)',
    6: 'Mod Chair',
    7: 'Captain\'s Chair',
    8: 'Chair (Simple)',
    9: 'Chippendale Table (3x3)',
    10: 'Bookshelf [Tall] (1x2)',
    11: 'Laptop',
    12: 'Microphone',
    13: 'Lucky Bamboo',
    14: 'Dining Chair (Square)'
}
mapitemname_to_id = {
    '地板': -1,
    'Whiteboard': 0,
    'Projector Screen': 1,
    'Chippendale Table (2x3)': 2,
    'TV (Flatscreen)': 3,
    'Bookshelf (2x4)': 4,
    'Potted Plant (Spikey)': 5,
    'Mod Chair': 6,
    'Captain\'s Chair': 7,
    'Chair (Simple)': 8,
    'Chippendale Table (3x3)': 9,
    'Bookshelf [Tall] (1x2)': 10,
    'Laptop': 11,
    'Microphone': 12,
    'Lucky Bamboo': 13,
    'Dining Chair (Square)': 14
}


def get_size_mapitem() -> int:
    return len(id_to_mapitemname)-1


def id_from_onehot(one_hot: RoomCell) -> int:
    for i in range(get_size_mapitem()):
        if one_hot[i]:
            return i
    return -1


def is_types(room_cell: RoomCell, types) -> bool:
    for i in range(get_size_mapitem()):
        if i in types:
            if room_cell[i] == 1:
                return True
    return False


def one_hot_mapitem(appear_prob: float) -> RoomCell:
    arr = [0] * get_size_mapitem()
    if random() >= appear_prob:
        arr.extend([0, 0])
        return arr
    arr[randint(0, get_size_mapitem() - 1)] = 1
    arr.extend(choice(orientations))
    return arr


def room_map_random(h: int, w: int, appear_prob: float) -> RoomMap:
    return [[one_hot_mapitem(appear_prob) for _ in range(w)] for _ in range(h)]


def room_map_empty(h: int, w: int) -> RoomMap:
    return [[one_hot_mapitem(0) for _ in range(w)] for _ in range(h)]


def random_population(size: int, h: int = H, w: int = W, appear_prob: float = P) -> Population:
    return [room_map_random(h, w, appear_prob) for _ in range(size)]


def fitness(room_map: RoomMap) -> Tuple[int, int, int, float]:
    from fitness import bias
    value = 0
    weight = 0

    for h in range(H):
        for w in range(W):
            _value, _weight = bias.fitness(room_map, h, w)
            value += _value
            weight += _weight

    diff = abs(value - weight)
    ratio = value / weight
    return value, weight, diff, ratio


def select_parents_pair(population: Population, fitness_func) -> Population:
    return choices(
        population=population,
        weights=[fitness_func(room_map)[3] for room_map in population],
        k=2
    )


def single_point_crossover(a: RoomMap, b: RoomMap) -> Tuple[RoomMap, RoomMap]:
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise ValueError("RoomMap should have the same (w, h)")
    new_a = a
    new_b = b
    new_a[H // 2:], new_b[H // 2:] = new_a[H // 2:], new_b[H // 2:]
    for i in range(H):
        new_a[i][W // 2:], new_b[i][W // 2:] = new_b[i][W // 2:], new_a[i][W // 2:]
    return new_a, new_b


def mutation(room_map: RoomMap, prob: float = M_P) -> None:
    if random() > prob:
        return
    ri = randint(0, H - 1)
    rj = randint(0, W - 1)
    # TODO: This heuristic might not be needed.
    if is_types(room_map[ri][rj], [-1]):
        room_map[ri][rj] = one_hot_mapitem(1)


def run_evo(
        populate_func,
        fitness_func,
        selection_func,
        crossover_func,
        mutation_func,
        generation_limit: int
) -> Tuple[Population, int]:
    population = populate_func(P_SIZE)

    i = 0
    for i in range(generation_limit):
        print(f"GEN={i + 1}, len(population)={len(population)}")

        population = sorted(
            population,
            key=lambda room_map: fitness_func(room_map)[3],
            reverse=True
        )

        print(f"Current best: (value={fitness_func(population[0])[0]}"
              f", weight={fitness_func(population[0])[1]})")

        if fitness_func(population[0])[1] > WEIGHT_LIMIT:
            print("Everyone exceed the weight_limit")
            break

        next_generation = population[:2]  # pick elites first

        for j in range(len(population)//2 - 1):
            parents = selection_func(population, fitness_func)
            c1, c2 = crossover_func(parents[0], parents[1])
            mutation_func(c1)
            mutation_func(c2)
            next_generation.extend([c1, c2])

        population = next_generation

    population = sorted(
        population,
        key=lambda room_map: fitness_func(room_map)[3],
        reverse=True
    )

    return population, i


if __name__ == '__main__':
    from prepare_data.workspace import workspace_population
    population, gens = run_evo(
        populate_func=workspace_population,
        fitness_func=fitness,
        selection_func=select_parents_pair,
        crossover_func=single_point_crossover,
        mutation_func=mutation,
        generation_limit=GEN_LIMIT
    )

    print(f"------------generation------------:\n{gens}")
    pprint(f"------------best population------------:\n")
    for h in range(H):
        for w in range(W):
            print(id_to_mapitemname[id_from_onehot(population[0][h][w])], end=" ")
        print("")
