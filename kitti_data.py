import torch
import torch.utils.data as data
import h5py 

class KITTI(data.Dataset):
    def __init__(self, datafile, sourcefile, nt):
        self.datafile = datafile
        self.sourcefile = sourcefile
        self.X = h5py.File(self.datafile, 'r')
        self.sources = h5py.File(self.sourcefile, 'r')
        self.nt = nt
        cur_loc = 0
        possible_starts = []

        my_array = self.X['data_0'][()]
        self.X = my_array 

        sources_array = self.sources['data_0'][()]
        self.sources = sources_array
        print('self.sources', self.sources)
        print('self.sources.shape', self.sources.shape)

        while cur_loc < self.X.shape[0] - self.nt + 1:
            if self.sources[cur_loc] == self.sources[cur_loc + self.nt - 1]:
                possible_starts.append(cur_loc)
                cur_loc += self.nt
            else:
                cur_loc += 1
        self.possible_starts = possible_starts

        print('self.possible_starts', self.possible_starts)
        print('len self.possible_starts', len(self.possible_starts))

    def __getitem__(self, index):
        loc = self.possible_starts[index]
        return self.X[loc:loc+self.nt]


    def __len__(self):
        return len(self.possible_starts)
