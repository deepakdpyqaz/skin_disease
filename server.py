import os
from glob import glob
import numpy as np
from messages.requests_pb2 import Input, Output
from distributed import Server
import logging
import requests
import os
import json
from operator import attrgetter
import base64

telegram = json.load(open(".credentials/telegram.json"))

TELEGRAM_TOKEN = telegram["token"]
TELEGRAM_CHAT_ID = telegram["chat_id"]
RESULTS = "results"


if not os.path.exists(RESULTS):
    os.mkdir(RESULTS)


def send_to_telegram(text):
    """
    Sends a text to telegram chat
    text: text to send
    """
    token = TELEGRAM_TOKEN
    chat_id = TELEGRAM_CHAT_ID
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        params = {"chat_id": chat_id, "text": text}
        response = requests.post(url, params=params)
        return response
    except Exception as e:
        print(e)
        return False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    filename="server.log",
    filemode="a",
)


class GlobalConfig:
    def __init__(self, filename="global_config.json"):
        self.filename = filename
        self.hasmap = {}
        if os.path.exists(self.filename):
            with open(self.filename, "r") as f:
                self.hasmap = json.load(f)
        self.persist()

    def persist(self):
        with open(self.filename, "w") as f:
            json.dump(self.hasmap, f)

    def __getitem__(self, key):
        key._task = ""
        key._id = 0
        encoded_key = base64.urlsafe_b64encode(key.SerializeToString()).decode()
        obj = self.hasmap[encoded_key]
        encoded_object = base64.urlsafe_b64decode(obj.encode())
        return Output.FromString(encoded_object)

    def __setitem__(self, key: Input, value: Output):
        key._task = ""
        key._id = 0
        value._id = 0
        self.hasmap[
            base64.urlsafe_b64encode(key.SerializeToString()).decode()
        ] = base64.urlsafe_b64encode(value.SerializeToString()).decode()
        self.persist()

    def __contains__(self, key: Input):
        key._task = ""
        key._id = 0
        return base64.urlsafe_b64encode(key.SerializeToString()).decode() in self.hasmap


class IterationConfig:
    def __init__(self):
        self.filename = "iteration.npy"
        self.iteration = np.array(0)
        if os.path.exists(self.filename):
            self.iteration = np.load(self.filename)
        self.persist()

    def persist(self):
        np.save(self.filename, self.iteration)

    def increment(self):
        self.iteration += 1
        self.persist()

    def decrement(self):
        self.iteration -= 1
        self.persist()


def convert_to_particle(solution, idx=0):
    index = int(np.round(solution[0] * (len(ranges["Batch Size"]) - 1)))
    batchSizeValue = ranges["Batch Size"][index]
    index = int(np.round(solution[1] * (len(ranges["amsgrad"]) - 1)))
    amsgrad = ranges["amsgrad"][index]
    index = int(np.round(solution[2] * (len(ranges["weight_decay"]) - 1)))
    weight_decay = ranges["weight_decay"][index]
    index = int(np.round(solution[3] * (len(ranges["lr"]) - 1)))
    lr = ranges["lr"][index]
    index = int(np.round(solution[4] * (len(ranges["b1"]) - 1)))
    b1 = ranges["b1"][index]
    index = int(np.round(solution[5] * (len(ranges["b2"]) - 1)))
    b2 = ranges["b2"][index]
    index = int(np.round(solution[6] * (len(ranges["epsilon"]) - 1)))
    epsilon = ranges["epsilon"][index]
    index = int(np.round(solution[7] * (len(ranges["factor"]) - 1)))
    factor = ranges["factor"][index]
    index = int(np.round(solution[8] * (len(ranges["patience"]) - 1)))
    patience = ranges["patience"][index]
    index = int(np.round(solution[9] * (len(ranges["cooldown"]) - 1)))
    cooldown = ranges["cooldown"][index]

    return Input(
        batch_size=batchSizeValue,
        amsgrad=amsgrad,
        weight_decay=weight_decay,
        lr=lr,
        b1=b1,
        b2=b2,
        epsilon=epsilon,
        factor=factor,
        patience=patience,
        cooldown=cooldown,
        _id=idx,
    )


def get_scores(population, name="iteration"):
    global global_hashmap, server
    scores = []
    results = []
    population_task = [convert_to_particle(p, idx) for idx, p in enumerate(population)]
    to_calculate = list()
    for p in population_task:
        if p in global_hashmap:
            results.append(global_hashmap[p])
        else:
            to_calculate.append(p)
    if to_calculate:
        server.create_task(name, to_calculate)
        calculated_results = server.get_results(timeout=2 * 60 * 60)
    else:
        calculated_results = []
    results.extend(calculated_results)
    scores = []
    losses = []
    train_loss = []
    train_accuracy = []
    for res in sorted(results, key=attrgetter("_id")):
        scores.append(res.score)
        losses.append(res.loss)
        train_loss.append(res.train_loss)
        train_accuracy.append(res.train_score)
    if len(scores) != len(population):
        logging.error(f"Lengths of scores and population do not match for {name}")
        send_to_telegram(f"Lengths of scores and population do not match for {name}")
        raise Exception("Lengths of scores and population do not match")
    # Updating the global hashmap
    for idx in range(len(population)):
        global_hashmap[population_task[idx]] = results[idx]
    return scores, losses, train_loss, train_accuracy


POPULATION_SIZE = 10  # Number of particles
NO_OF_ITERATIONS = 30  # Number of iterations of MRFO
LOWER_BOUND = 0.0
UPPER_BOUND = 1.0
finished_iterations = IterationConfig()

ranges = {
    "Batch Size": [8, 16, 32],
    "amsgrad": [True, False],
    "weight_decay": [None, 1e-5, 1e-6],
    "lr": np.geomspace(1e-3, 1e-5, 3),
    "b1": np.arange(0.8, 0.96, 0.05),
    "b2": np.arange(0.990, 0.9991, 0.003),
    "epsilon": [1e-7, 1e-8],
    "factor": [0.1, 0.2, 0.5],
    "patience": [2, 3, 5],
    "cooldown": [2, 3, 5],
}


server = Server(name="mrfo_btp")
global_hashmap = GlobalConfig()
best_score = np.array(0.0)
best_solution = None
SOLUTION_SIZE = len(ranges.keys())

population = np.random.uniform(
    low=LOWER_BOUND, high=UPPER_BOUND, size=(POPULATION_SIZE, SOLUTION_SIZE)
)
logging.info("Code started")
send_to_telegram("Code started")
print("Code started")

if (
    os.path.exists("best_score.npy")
    and os.path.exists("best_solution.npy")
    and os.path.exists("population.npy")
):
    # Load current state
    population = np.load("population.npy")
    best_solution = np.load("best_solution.npy")
    best_score = np.load("best_score.npy")
else:
    # First iteration
    scores, losses, train_loss, train_accuracy = get_scores(population, "Init Iteration")
    best_score = np.max(scores)
    best_solution = population[np.argmax(scores)]


logging.info(
    f"Starting MRFO with {finished_iterations.iteration} iterations completed and {best_score} as best score"
)
send_to_telegram(
    f"Starting MRFO with {finished_iterations.iteration} iterations completed and {best_score} as best score"
)
print(
    f"Starting MRFO with {finished_iterations.iteration} iterations completed and {best_score} as best score"
)
for iterationNumber in range(finished_iterations.iteration, NO_OF_ITERATIONS):
    r = np.random.uniform(1)
    alpha = 2 * r * np.sqrt(np.abs(np.log(r)))
    r1 = np.random.uniform(1)
    beta = (
        2
        * np.exp(r1 * (NO_OF_ITERATIONS - iterationNumber + 1) / NO_OF_ITERATIONS)
        * np.sin(2 * np.pi * r1)
    )
    s = 2
    r2 = np.random.uniform(1)
    r3 = np.random.uniform(1)
    if np.random.uniform() < 0.5:  # Cyclone forgaging
        if iterationNumber / NO_OF_ITERATIONS < np.random.rand():  # Exploratory
            x_rand = np.random.uniform(size=SOLUTION_SIZE)
            population[0] = (
                x_rand + r * (x_rand - population[0]) + beta * (x_rand - population[0])
            )
            population[0] = np.clip(population[0], LOWER_BOUND, UPPER_BOUND)
            for i in range(1, POPULATION_SIZE):
                population[i] = (
                    x_rand
                    + r * (population[i - 1] - population[i])
                    + beta * (x_rand - population[i])
                )
                population[i] = np.clip(population[i], LOWER_BOUND, UPPER_BOUND)
        else:  # Exploitative
            population[0] = (
                best_solution
                + r * (best_solution - population[0])
                + beta * (best_solution - population[0])
            )
            population[0] = np.clip(population[0], LOWER_BOUND, UPPER_BOUND)
            for i in range(1, POPULATION_SIZE):
                population[i] = (
                    best_solution
                    + r * (population[i - 1] - population[i])
                    + beta * (best_solution - population[i])
                )
                population[i] = np.clip(population[i], LOWER_BOUND, UPPER_BOUND)

    else:  # Chain forgaging
        population[0] = (
            population[0]
            + r * (best_solution - population[0])
            + alpha * (best_solution - population[0])
        )
        population[0] = np.clip(population[0], LOWER_BOUND, UPPER_BOUND)
        for i in range(1, POPULATION_SIZE):
            population[i] = (
                population[i]
                + r * (population[i - 1] - population[i])
                + alpha * (best_solution - population[i])
            )
            population[i] = np.clip(population[i], LOWER_BOUND, UPPER_BOUND)

    # Calculating Fitness
    scores, losses, train_loss, train_accuracy = get_scores(population, f"Iteration-{iterationNumber}-1")
    if np.max(scores) > best_score:
        best_score = np.max(scores)
        best_solution = population[np.argmax(scores)]

    # Summersault forgaging
    for i in range(POPULATION_SIZE):
        population[i] = population[i] + s * (r2 * best_solution - r3 * population[i])
        population[i] = np.clip(population[i], LOWER_BOUND, UPPER_BOUND)

    # Calculating Fitness
    scores, losses, train_loss, train_accuracy = get_scores(population, f"Iteration-{iterationNumber}-2")
    if np.max(scores) > best_score:
        best_score = np.max(scores)
        best_solution = population[np.argmax(scores)]
    np.save("population.npy", population)
    np.save("best_solution.npy", best_solution)
    np.save("best_score.npy", best_score)
    iteration_dir = os.path.join(
        RESULTS, f"iteration-{iterationNumber}-{str(best_score).replace('.','_')}"
    )
    if not os.path.exists(iteration_dir):
        os.makedirs(iteration_dir)
    np.save(os.path.join(iteration_dir, "population.npy"), population)
    np.save(os.path.join(iteration_dir, "best_solution.npy"), best_solution)
    np.save(os.path.join(iteration_dir, "best_score.npy"), best_score)
    np.save(os.path.join(iteration_dir, "scores.npy"), scores)
    np.save(os.path.join(iteration_dir, "losses.npy"), losses)
    np.save(os.path.join(iteration_dir, "train_loss.npy"), train_loss)
    np.save(os.path.join(iteration_dir, "train_accuracy.npy"), train_accuracy)

    finished_iterations.increment()
    logging.info(f"Iteration {iterationNumber} completed with best score {best_score}")
    send_to_telegram(
        f"Iteration {iterationNumber} completed with best score {best_score}"
    )
    print(f"Iteration {iterationNumber} completed with best score {best_score}")


logging.info(f"MRFO completed with best score {best_score}")
send_to_telegram(f"MRFO completed with best score {best_score}")
print(f"MRFO completed with best score {best_score}")
