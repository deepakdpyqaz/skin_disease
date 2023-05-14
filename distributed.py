import pika
import json
import time
import uuid
from messages.requests_pb2 import Input, Output
import pyrebase
from operator import attrgetter
import datetime
import requests
from multiprocessing import Queue
import threading
import psutil
from stats import get_system_stats
import traceback
import warnings

telegram = json.load(open(".credentials/telegram.json"))

TELEGRAM_TOKEN = telegram["token"]
TELEGRAM_CHAT_ID = telegram["chat_id"]


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


def send_to_telegram_document(document):
    """
    Sends a document to telegram chat
    document: path to file
    """
    token = TELEGRAM_TOKEN
    chat_id = TELEGRAM_CHAT_ID
    url = f"https://api.telegram.org/bot{token}/sendDocument"
    params = {"chat_id": chat_id}
    post_files = {"document": open(document, "rb")}
    header_list = {
        "Accept": "*/*",
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/111.0",
    }
    try:
        response = requests.post(
            url, params=params, data="", headers=header_list, files=post_files
        )
        return response
    except Exception as e:
        print(e)
        return False


def loggworker(queue, db=None, telegram=True):
    print("Logging started")
    while True:
        data, log_type = queue.get()
        if data == None:
            break
        try:
            if log_type == "start":
                db.child("tasks").child(data.get("task")).child("client").update(
                    {data.get("client"): data.get("start")}
                )
            else:
                print("Log:", str(data))
                if db:
                    db.child("tasks").child(data.get("task")).child("results").push(
                        data
                    )
                if telegram:
                    send_to_telegram(str(data))
        except Exception as e:
            pass


def make_heartbeat(client, db, stop, timer=10 * 60):
    try:
        while stop.is_set() == False:
            stats = get_system_stats()
            stats["timestamp"] = datetime.datetime.utcnow().strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )
            stats["timer"] = timer
            db.child("heartbeat").child(client).set(stats)
            time.sleep(timer)
    except:
        traceback.print_exc()


class Server:
    def __init__(
        self,
        name="server",
        firebase_config_path=".credentials/firebase.json",
        pika_config_path=".credentials/rabbitmq.json",
    ):
        if name == "server":
            suffix = str(uuid.uuid4())[:8]
            self.name = f"{name}_{suffix}"
        else:
            self.name = name
        firebase_config = json.load(open(firebase_config_path))
        pika_credentials = json.load(open(pika_config_path))
        self.firebase = pyrebase.initialize_app(firebase_config)
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=pika_credentials["url"],
                virtual_host=pika_credentials["vhost"],
                port=pika_credentials["port"],
                heartbeat=0,
                blocked_connection_timeout=None,
                socket_timeout=None,
                credentials=pika.PlainCredentials(
                    pika_credentials["username"], pika_credentials["password"]
                ),
            )
        )

        self.channel = self.connection.channel()
        self.input_queue = f"{self.name}_input"
        self.output_queue = f"{self.name}_output"
        self.channel.queue_declare(queue=self.input_queue, durable=True)
        self.channel.queue_declare(queue=self.output_queue, durable=True)
        self.tasks_assigned = 0

    def __del__(self):
        try:
            self.connection.close()
        except:
            pass

    def create_task(self, task_name, tasks):
        if len(tasks) > 500:
            raise Exception("Too many tasks")
        self.task = (
            self.firebase.database()
            .child("tasks")
            .push(
                {
                    "initiator": self.name,
                    "name": task_name,
                    "items": len(tasks),
                    "start": datetime.datetime.utcnow().strftime(
                        "%Y-%m-%d %H:%M:%S.%f"
                    ),
                }
            )
        )
        print(f"Created task {self.task['name']}")
        for id, task in enumerate(tasks):
            if task._id == 0:
                task._id = id + 1
            task._task = self.task["name"]
            self.channel.basic_publish(
                exchange="", routing_key=self.input_queue, body=task.SerializeToString()
            )
        self.tasks_assigned = len(tasks)

    def get_results(self, timeout=10 * 60):
        results = []
        for method, properties, body in self.channel.consume(
            self.output_queue, inactivity_timeout=timeout
        ):
            if body == None:
                break
            results.append(Output().FromString(body))
            self.channel.basic_ack(delivery_tag=method.delivery_tag)
            self.tasks_assigned -= 1
            print(f"Tasks left: {self.tasks_assigned}")
            if self.tasks_assigned == 0:
                break
        if self.tasks_assigned > 0:
            warnings.Warn("Some tasks were not completed")

        self.tasks_assigned = 0

        self.firebase.database().child("tasks").child(self.task["name"]).update(
            {"end": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")}
        )
        return sorted(results, key=attrgetter("_id"))


class Client:
    def __init__(
        self,
        server_name,
        func,
        name="client",
        firebase_config_path=".credentials/firebase.json",
        pika_config_path=".credentials/rabbitmq.json",
        heartbeat=10 * 60,
    ):
        if name == "client":
            suffix = str(uuid.uuid4())[:8]
            self.name = f"{name}_{suffix}"
        else:
            self.name = name
        pika_credentials = json.load(open(pika_config_path))
        firebase_config = json.load(open(firebase_config_path))
        self.firebase = pyrebase.initialize_app(firebase_config)
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=pika_credentials["url"],
                virtual_host=pika_credentials["vhost"],
                port=pika_credentials["port"],
                heartbeat=0,
                blocked_connection_timeout=None,
                socket_timeout=None,
                credentials=pika.PlainCredentials(
                    pika_credentials["username"], pika_credentials["password"]
                ),
            )
        )
        self.channel = self.connection.channel()
        self.channel.basic_qos(prefetch_count=1)
        self.input_queue = f"{server_name}_input"
        self.output_queue = f"{server_name}_output"
        self.channel.queue_declare(queue=self.input_queue, durable=True)
        self.channel.queue_declare(queue=self.output_queue, durable=True)
        self.func = func
        self.log_queue = Queue()
        self.logger = None
        if heartbeat > 0:
            self.stop_heartbeat = threading.Event()
            self.heartbeat = threading.Thread(
                target=make_heartbeat,
                args=(self.name, self.firebase.database(), self.stop_heartbeat),
                kwargs={"timer": heartbeat},
            )
            self.heartbeat.start()
        else:
            self.heartbeat = None

    def __del__(self):
        try:
            if self.heartbeat:
                self.stop_heartbeat.set()
            self.heartbeat.join()
            self.logger.join()
            self.connection.close()
        except:
            pass

    def make_log(self, log, log_type=None):
        self.log_queue.put((log, log_type))

    def start_process(self, timeout=10 * 60):
        print(f"Started {self.name} on queue {self.input_queue}")
        self.logger = threading.Thread(
            target=loggworker, args=(self.log_queue, self.firebase.database())
        )
        self.logger.start()
        for method, properties, body in self.channel.consume(
            self.input_queue, inactivity_timeout=timeout
        ):
            if body == None:
                break
            try:
                start = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")
                ip = Input().FromString(body)
                task_log = {
                    "start": start,
                    "client": self.name,
                    "task": ip._task,
                    "id": ip._id,
                }
                self.make_log(task_log, "start")
                op = self.func(ip)
                op._id = ip._id
                self.channel.basic_publish(
                    exchange="",
                    routing_key=self.output_queue,
                    body=op.SerializeToString(),
                )
                self.channel.basic_ack(delivery_tag=method.delivery_tag)
                end = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")
                task_log["end"] = end
                self.make_log(task_log)
            except Exception as e:
                print(e)
                self.channel.basic_nack(delivery_tag=method.delivery_tag)
                continue
        self.log_queue.put((None, None))
        self.logger.join()
        if self.heartbeat:
            self.stop_heartbeat.set()
            self.heartbeat.join()
