from utils import VideoLoader
from utils import Formatter
import numpy as np
import sys
import json

def main(option):
    conf = json.load(open('conf/sti.json'))[option]
    loader = VideoLoader(conf['video_path'], conf['offset'])
    w = conf['roi']['w2'] - conf['roi']['w1']
    h = conf['roi']['h2'] - conf['roi']['h1']

    # Set formatter
    im = loader.read()
    formatter = Formatter(im.shape,
                         -25,
                         1.0,
                         conf['roi']['w1'],
                         conf['roi']['w2'],
                         conf['roi']['h1'],
                         conf['roi']['h2'])
    # im = formatter.apply(im)
    np.save('tmp.npy', im)

if __name__ == "__main__":
    option = 'd1'
    if len(sys.argv) > 1:
        option = sys.argv[1]
    main(option)
