from enum import Enum
import math


class MachineConfig(object):
    idx = 0

    @staticmethod
    def reset_idx():
        MachineConfig.idx = 0

    def __init__(self, index,cpu_capacity, memory_capacity, disk_capacity, name="", processing=None,cpu=None, memory=None, disk=None):
        self.reset_idx()
        self.processing=processing
        self.cpu_capacity = cpu_capacity
        self.memory_capacity = memory_capacity
        self.disk_capacity = disk_capacity
        self.name = name
        self.cpu = cpu_capacity if cpu is None else cpu
        self.memory = memory_capacity if memory is None else memory
        self.disk = disk_capacity if disk is None else disk
        self.index=index
        self.id = MachineConfig.idx

        MachineConfig.idx += 1


class MachineDoor(Enum):
    TASK_IN = 0
    TASK_OUT = 1
    NULL = 3


class Machine(object):
    def __init__(self, machine_config, idx=None):
        self.id = idx if idx is not None else MachineConfig.idx
        self.processing=machine_config.processing
        self.index=machine_config.index

        self.cpu_capacity = machine_config.cpu_capacity
        self.memory_capacity = machine_config.memory_capacity
        self.disk_capacity = machine_config.disk_capacity
        self.cpu = machine_config.cpu
        self.memory = machine_config.memory
        self.disk = machine_config.disk
        self.name = machine_config.name
        self.powerlist=[]
        if self.name =='PowerEdgeC5220' :
            self.powerlist = [71.0, 77.9, 83.4, 89.2, 95.6, 102.0, 108.0, 114.0, 119.0, 123.0, 126.0]
        elif self.name=='HPProLiantDL2000':
            self.powerlist = [81.4, 110, 125, 139, 153, 167, 182, 199, 214, 229, 244]
        elif self.name == 'IBMSystemx3630M4':
            self.powerlist =[0.45, 0.78, 0.90, 1.03, 1.65, 1.66, 1.69, 1.71, 1.72, 1.74, 1.75]
        elif self.name == 'IBMSystemxiDataPlexdx360M3':
            self.powerlist = [68.7, 78.3, 84.0, 88.4, 92.5, 97.3, 104.0, 111.0, 121.0, 131.0, 137.0]
        elif self.name=='XeonE502673':
            self.powerlist = [75.2, 78.2, 84.1, 89.6, 94.9, 100.0, 105.0, 109.0, 112.0, 115.0, 117.0]
        elif self.name == 'Systemx3200M3':
            self.powerlist = [68.7, 78.3, 84.0, 88.4, 92.5, 97.3, 104.0, 111.0, 121.0, 131.0, 137.0]

        #
        elif self.name == 'Dell-PowerEdgeHS5610':
            self.powerlist = [122, 220, 245, 268, 295, 325, 360, 448, 535, 600, 602]

        #https://www.spec.org/power_ssj2008/results/res2024q1/power_ssj2008-20231205-01347.html
        elif self.name == 'ASUSTEK-RS720A-E12-RS12':
            self.powerlist = [117, 231, 278, 319, 353, 382, 412, 461,519, 591, 685]

        #https://www.spec.org/power_ssj2008/results/res2024q1/power_ssj2008-20240130-01363.html
        elif self.name == 'ThinkEdgeSE360V2':
            self.powerlist = [52.4, 67.7, 73.4, 78.5, 83.0, 88.5, 94.6, 104, 112, 123, 139]

        #https://www.spec.org/power_ssj2008/results/res2024q1/power_ssj2008-20231205-01349.html
        elif self.name == 'QuantaGridD54Q-2U':
            self.powerlist = [152, 284, 328, 373, 421, 476, 538, 626, 838, 909, 968]

        #https://www.spec.org/power_ssj2008/results/res2024q1/power_ssj2008-20231120-01339.html
        elif self.name == 'SuperServerSYS-521E-WR':
            self.powerlist = [145, 175, 214, 231, 264, 295, 332, 371, 416, 485, 564]

        #https://www.spec.org/power_ssj2008/results/res2024q1/power_ssj2008-20240227-01376.html
        elif self.name == 'Dell-PowerEdgeHS5610':
            self.powerlist = [122, 220, 245, 268, 295, 325, 360, 448, 535, 600, 602]

        #https://www.spec.org/power_ssj2008/results/res2024q1/power_ssj2008-20240130-01364.html
        elif self.name == 'FujitsuRX1440M2':
            self.powerlist = [67.1, 132, 160, 187, 212, 236, 259, 288, 328, 364, 391]

        #https://www.spec.org/power_ssj2008/results/res2024q1/power_ssj2008-20231205-01345.html
        elif self.name == 'FusionServer-1288H-V7':
            self.powerlist = [204, 252, 291, 328, 368, 409, 453, 503, 569, 720, 817]

        #https://www.spec.org/power_ssj2008/results/res2024q1/power_ssj2008-20240208-01367.html
        elif self.name == 'ThinkSystem-SR850-V3':
            self.powerlist = [454, 569, 663, 756, 847, 949, 1049, 1161, 1461, 1873, 1923]


        #https://www.spec.org/power_ssj2008/results/res2024q1/power_ssj2008-20231121-01341.html
        elif self.name == 'HPProLiant-ML30-Gen11':
            self.powerlist = [22.6, 32.5, 40.4, 47.6, 54.4, 61.3, 69.2, 76.5, 83.7, 91.8, 98.2]

        #https://www.spec.org/power_ssj2008/results/res2024q1/power_ssj2008-20231104-01332.html
        elif self.name == 'FALINUX-AnyStor-700EC-NM':
            self.powerlist = [53.9, 64.1, 71.3, 78.1, 84.6, 91.5, 98.6, 106, 114, 125, 140]






            # cpu consumption in 100
        self.power_consumed=0
        self.cluster = None
        self.task_instances = []
        self.machine_door = MachineDoor.NULL
        self.is_working = True  # Flag to indicate if machine is operational
        if idx is None:
            MachineConfig.idx += 1
    @property
    def max_power(self):
        return self.powerlist[-1]

    @property
    def min_power(self):
        return self.powerlist[0]

    def power(self):
        cpu =self.cpu_capacity-self.cpu
        # print('cpu_capacity',self.cpu_capacity)
        # print('self.cpu',self.cpu)
        # print('cpu before',cpu)

        index = math.floor(cpu * 10/self.cpu_capacity)
        cpu = max(0, min(cpu, self.cpu_capacity))
        # print('cpu after',cpu)
        #print('cpu_capacity',self.cpu_capacity)
        #print('index',index)

        left = self.powerlist[index]
        #print(cpu % 10/self.cpu_capacity)
        right = self.powerlist[index + 1 if (cpu % 10/self.cpu_capacity != 0 and index != 10) else index]
        alpha = (cpu / 10/self.cpu_capacity) - index
        return alpha * right + (1 - alpha) * left

    def powerfitness(self,taskscpu):
        cpu =self.cpu+taskscpu
        # if cpu<0:
        #     return None
        #print('cpu_capacity',self.cpu_capacity)
        #print('self.cpu',self.cpu)
        cpu = max(0, min(cpu, self.cpu_capacity))

        index = math.floor(cpu * 10/self.cpu_capacity)


        left = self.powerlist[index]
        right = self.powerlist[index + 1 if int(cpu % 10/self.cpu_capacity) != 0 else index]
        alpha = (cpu / 10/self.cpu_capacity) - index
        return alpha * right + (1 - alpha) * left



    def breakdown(self):
        self.is_working = False  # Set machine as not working
    #
    def repair(self):
        self.is_working = True  # Set machine as operational again



    def get_power_consumed(self,task_instance):
        self.power_previous = self.power()
        self.cpu_=self.cpu

        self.cpu_ -= task_instance.cpu

        self.cpu_=max(self.cpu_, 0)

        self.power_next = self.powerfitness(task_instance.cpu)

        self.power_consumed=self.power_next-self.power_previous
        return self.power_consumed


    def run_task_instance(self, task_instance):
        self.cpu -= task_instance.cpu

        self.memory -= task_instance.memory
        self.disk -= task_instance.disk
        self.task_instances.append(task_instance)
        self.machine_door = MachineDoor.TASK_IN


    def simulate_run_task_instance(self, task_instance):
        self.power_previous = self.power()

        self.cpu -= task_instance.cpu
        self.power_next = self.power()
        self.power_consumed=self.power_next-self.power_previous

        self.memory -= task_instance.memory
        self.disk -= task_instance.disk
        self.task_instances.append(task_instance)
        self.machine_door = MachineDoor.TASK_IN


    def stop_task_instance(self, task_instance):
        self.cpu += task_instance.cpu
        self.memory += task_instance.memory
        self.disk += task_instance.disk
        self.machine_door = MachineDoor.TASK_OUT

    @property
    def running_task_instances(self):
        ls = []
        for task_instance in self.task_instances:
            if task_instance.started and not task_instance.finished:
                ls.append(task_instance)
        return ls

    @property
    def finished_task_instances(self):
        ls = []
        for task_instance in self.task_instances:
            if task_instance.finished:
                ls.append(task_instance)
        return ls

    def attach(self, cluster):
        self.cluster = cluster

    def accommodate(self, task):
        return self.cpu >= task.task_config.cpu and \
               self.memory >= task.task_config.memory and \
               self.disk >= task.task_config.disk

    @property
    def feature(self):
        return [self.cpu, self.memory, self.disk,self.processing]

    @property
    def capacity(self):
        return [self.cpu_capacity, self.memory_capacity, self.disk_capacity]

    @property
    def state(self):
        return {
            'id': self.id,
            'cpu_capacity': self.cpu_capacity,
            'memory_capacity': self.memory_capacity,
            'disk_capacity': self.disk_capacity,
            'cpu': self.cpu / self.cpu_capacity,
            'memory': self.memory / self.memory_capacity,
            'disk': self.disk / self.disk_capacity,
            'running_task_instances': len(self.running_task_instances),
            'finished_task_instances': len(self.finished_task_instances)
        }

    def __eq__(self, other):
        return isinstance(other, Machine) and other.id == self.id
