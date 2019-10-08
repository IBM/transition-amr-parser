class StackRNN(object):
    def __init__(self, cell, initial_state, dropout, activation, empty_embedding=None):
        self.cell = cell
        self.dropout = dropout
        self.states = [initial_state]
        self.embeddings = [None]
        self.strings = [None]
        self.empty = None
        self.activation = activation
        if empty_embedding is not None:
            self.empty = empty_embedding

    def push(self, embedding, string):
        h, c = self.states[-1]
        output = self.cell(embedding, (self.dropout(h), c))
        self.states.append(output)
        self.embeddings.append(embedding)
        self.strings.append(string)

    def pop(self):
        self.states.pop()
        return (self.embeddings.pop(), self.strings.pop())

    def last(self):
        return (self.embeddings[-1], self.strings[-1])

    def output(self):
        return self.activation(self.states[-1][0]) if len(self.states) > 1 else self.empty

    def clear(self):
        while len(self.states) > 1:
            self.states.pop()
            self.embeddings.pop()
            self.strings.pop()

    def __len__(self):
        return len(self.states) - 1

    def __str__(self):
        return '<StackRNN>: '+' '.join(str(t) for t in self.strings[1:])
