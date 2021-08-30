from .config import cfg
from .utils import get_tvm_module_N_params

import tvm
from tvm import relay, auto_scheduler
from tvm.relay import data_dep_optimization as ddo

from argparse import ArgumentParser

def run_tuning(tasks, task_weights, json_file, trials=1000, use_sparse=False):
    print("Begin tuning...")
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=trials,  # change this to 20000 to achieve the best performance
        runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
        measure_callbacks=[auto_scheduler.RecordToFile(json_file)],
    )

    if use_sparse:
        from tvm.topi.sparse.utils import sparse_sketch_rules

        search_policy = [
            auto_scheduler.SketchPolicy(
                task,
                program_cost_model=auto_scheduler.XGBModel(),
                init_search_callbacks=sparse_sketch_rules(),
            )
            for task in tasks
        ]

        tuner.tune(tune_option, search_policy=search_policy)
    else:
        tuner.tune(tune_option)


# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.

def make_parser():
    parser = ArgumentParser(
        description=f"usage ./{__file__} ")    
    return parser
    
if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    print("Get module...")
    mod, params = get_tvm_module_N_params(
        cfg.model_path, 
        input_name=cfg.input_name,
        batch_size=cfg.batch_size,
        input_shape=cfg.input_shape,
        layout=cfg.layout,
        dtype=cfg.dtype,
        use_sparse=cfg.use_sparse,
    )


    print("Extract tasks...")
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, cfg.target)

    for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        print(task.compute_dag)

    run_tuning(tasks, task_weights, cfg.json_file, trials=200)