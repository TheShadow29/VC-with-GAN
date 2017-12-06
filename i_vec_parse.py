import pickle
import re
import numpy as np

if __name__ == '__main__':
    with open('./data/all_ivectors.txt', 'r') as f:
        lines = f.read()

    pat = re.compile(r'(\w*)_\d*\s\[(.*)\]')
    all_tuples = pat.findall(lines)
    i_vec_dict = dict()

    for k, v in all_tuples:
        i_vec_dict[k] = np.array(v.split(), dtype=np.float32)

    out_f = open('i_vec_dict.pkl', 'wb')
    pickle.dump(i_vec_dict, out_f)
    out_f.close()
