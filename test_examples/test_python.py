import torch.utils.data


class Box(torch.utils.data.Dataset):
    def __init__(self, boxname, size, color):
        self.boxname = boxname
        self.size = size
        self.color = color  # self就是用于存储对象属性的集合，就算没有属性self也是必备的
        self.__indices__ = None

    def open(self, myself):
        print('-->用自己的myself，打开那个%s,%s的%s' % (myself.color, myself.size, myself.boxname))
        print('-->用类自己的self，打开那个%s,%s的%s' % (self.color, self.size, self.boxname))

    def close(self):
        print('-->关闭%s，谢谢' % self.boxname)

    def indices(self):
        if self.__indices__ is not None:
            return self.__indices__
        else:
            return range(len(self))

    @property
    def num_node_features(self):
        r"""Returns the number of features per node in the dataset."""
        return self[0].boxname

    def __len__(self):
        r"""The number of examples in the dataset."""
        if self.__indices__ is not None:
            return len(self.__indices__)
        return self.len()


b = Box('魔盒', '14m', '红色')
indices = b.indices()
b.num_node_features()
pass