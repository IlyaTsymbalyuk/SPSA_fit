import os
import requests
import subprocess
import sys
from threading import Thread
from time import sleep
subprocess.check_call([sys.executable, "-m", "pip", "install", "gputil", "psutil", "--upgrade"])
import GPUtil
import psutil
from datetime import datetime
import json

class ColabMonitor():
    _interval = 1
    _isLooping = False
    _killDaemon = False
    
    tpu = None
    def update(self):
        try:
            gpu = GPUtil.getGPUs()[0]
        except:
            gpu = None
        loadavg = 3.89 # psutil.getloadavg()[1]
        #disk_counter = psutil.disk_io_counters()
        net_counter = psutil.net_io_counters()

        self.payload['5m_loadavg'].append(loadavg)
        self.payload['cpus_load'].append(psutil.cpu_percent(percpu=True))
        self.payload['virt_mem'].append(psutil.virtual_memory().percent / 100)
        self.payload['disk_usage'].append(psutil.disk_usage(self.cwd).percent / 100)
        self.payload['net_sent'].append((net_counter.bytes_sent - self._last_bytes_sent) / 1048576)
        self.payload['net_recv'].append((net_counter.bytes_recv - self._last_bytes_recv) / 1048576)
    
        self._last_bytes_sent = net_counter.bytes_sent
        self._last_bytes_recv = net_counter.bytes_recv
        if gpu is not None:
            self.payload['gpu_load'].append(gpu.load * 100)
            self.payload['gpu_mem'].append(gpu.memoryUtil)
        if self.tpu is not None:
            self.payload['tpu_idle'].append(self.tpu.idle.value)
            self.payload['tpu_mxu'].append(self.tpu.mxu.value)

        self.save()

    def save(self):
        with open('./monitor_{}.json'.format(self.now), 'w') as outfile:
            json.dump(self.payload, outfile)

    def __init__(self, tpu=None):
        self.cwd = os.getcwd()
        self.now  = datetime.now().strftime("%d%m%Y_%H%M%S")

        self.payload = {
            '5m_loadavg': [],
            'cpus_load': [],
            'virt_mem': [],
            'disk_usage': [],
            #'disk-counter': [disk_counter.read_bytes, disk_counter.write_bytes],
            'net_sent': [],
            'net_recv': [],
            'gpu_load': [],
            'gpu_mem': [],
            'tpu_idle': [],
            'tpu_mxu': []
        }

        self.payload['total_virt_mem'] = psutil.virtual_memory().total / 1048576
        self.payload['interval'] = self._interval

        try:
            gpu = GPUtil.getGPUs()[0]
            self.payload['total_gpu_mem'] = gpu.memoryTotal
            self.payload['gpu_name'] = gpu.name
        except:
            pass
        if tpu is not None:
            self.tpu = self.Tensorflow_TPUMonitor(tpu, self)
            self.payload['tpu_type'] = self.tpu.type_n_cores
        self.payload['total_disk_space'] = psutil.disk_usage(self.cwd).total / 1048576
        net_counter = psutil.net_io_counters()
        self._last_bytes_sent = net_counter.bytes_sent
        self._last_bytes_recv = net_counter.bytes_recv

    def loop(self):
        while self._isLooping:
            self.update()
            sleep(self._interval)

    def start(self):
        if (self._isLooping):
            raise Exception("Monitoring already started!")
        thread = Thread(target=self.loop)
        self._isLooping = True
        thread.start()
        if self.tpu is not None:
            self.tpu.start()
        return self

    def setInterval(self, interval_s):
        self._interval = interval_s
        return self

    def stop(self):
        self._isLooping = False
        if self.tpu is not None:
            self.tpu.stop()
        return self

    class Tensorflow_TPUMonitor():
        def __init__(self, tpu, colabMonitor):
            from tensorflow.python.profiler.internal import _pywrap_profiler
            from multiprocessing import Value
            service_addr = tpu.get_master()
            self.service_addr = service_addr.replace('grpc://', '').replace(':8470', ':8466')
            self.monitor = _pywrap_profiler.monitor
            self.colabMonitor = colabMonitor
            self.mxu = Value('d', 0)
            self.idle = Value('d', 100)
            ret = self.monitor(self.service_addr, 1, 2, False)
            for line in ret.strip().split("\n"):
                if line.startswith("TPU type:"):
                    self.type_n_cores = line[10:] + "-" + str(tpu.num_accelerators()['TPU'])
                    break
            self.exit_loop = None

        def start(self):
            if self.exit_loop is not None and not self.exit_loop.is_set():
                raise Exception("TPU monitoring already started")
            from multiprocessing import Event, Process
            self.exit_loop = Event()
            self.process_loop = Process(target=self.loop)
            self.process_loop.start()

        def loop(self):
            while not self.exit_loop.is_set():
                self.update(self.colabMonitor._interval - 1)
                sleep(1)

        def update(self, interval_s):
            ret = self.monitor(self.service_addr, interval_s * 1000, 2, False)
            self.idle.value = 100
            for line in ret.split("\n"):
                line = line.strip()
                if line.startswith("TPU idle time"):
                    self.idle.value = float(line[33:-1])
                elif line.startswith("Utilization of"):
                    self.mxu.value = float(line[52:-1])

        def stop(self):
            self.exit_loop.set()
            self.process_loop.join()