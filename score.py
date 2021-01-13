
import sys

from datio import IO_manager
from model_IO import model_io
from model.count_manager import count_manager


def main(run_dir, model_name, model_dir, data_path, output_path):

    mio = model_io(run_dir, model_name)
    model = mio.load_model(model_dir)
    data, n_worlds, n_vars = IO_manager.read_from_csv(data_path, ',')
    cmgr = count_manager()
    data_name="data"
    data = cmgr.compress_data_set(data,data_name)
    model.set_count_manager(cmgr)
    ll = model.LL(data,data_name)
    size = model.sdd_size()

    with open(output_path,'w+') as f:
        print(f"Size: {size}", file=f)
        print(f"LL: {ll}", file=f)




if __name__ == "__main__":
    run_dir = sys.argv[1]
    model_name = sys.argv[2]
    model_dir = sys.argv[3]
    data_path = sys.argv[4]
    output_path = sys.argv[5]
    main(run_dir, model_name, model_dir, data_path, output_path)