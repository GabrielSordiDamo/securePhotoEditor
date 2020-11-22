
class Stack:

    def __init__(self, arrayLenght):
        self.array = []
        self.arrayLenght = arrayLenght

    def peek(self):
        if not self.isEmpty():
            return self.array[-1]
        else:
            return -1

    def pop(self):
        if not self.isEmpty():
            return self.array.pop()
        else:
            return -1

    def push(self, value):
        if len(self.array) < self.arrayLenght:
            self.array.append(value)
        else:
            return -1

    def isEmpty(self):
        return True if len(self.array) == 0 else False

    def __len__(self):
        return len(self.array)

    def __str__(self):
        return str(self.array)
