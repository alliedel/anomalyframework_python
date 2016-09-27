
class AttrDict(dict):
    def __init__(self, **kwargs):
        super(AttrDict, self).__init__(**kwargs)
        self.__dict__ = self

