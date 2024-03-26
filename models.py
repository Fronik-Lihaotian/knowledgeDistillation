from torch import nn
from model_v2 import MobileNetV2
from model_s import mobilenet_s_medium, mobilenet_s_small


class Teacher(nn.Module):
    def __init__(self, num_class=1000):
        super(Teacher, self).__init__()
        self.teacher = MobileNetV2(num_classes=num_class)

    def forward(self, x):
        return self.teacher(x)


class Student(nn.Module):
    def __init__(self, num_class=1000):
        super(Student, self).__init__()
        self.student = mobilenet_s_medium(num_class=num_class)

    def forward(self, x):
        return self.student(x)


