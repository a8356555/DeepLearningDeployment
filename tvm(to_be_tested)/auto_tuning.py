from .config import cfg
from .utils import get_tvm_module_N_params
import tvm
from tvm import relay, autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner

from argparse import ArgumentParser

# You can skip the implementation of this function for this tutorial.
def tune_kernels(
    tasks, 
    measure_option, 
    tuner="gridsearch", 
    early_stopping=None, 
    log_filename="tuning.log"
):

    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(task, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(task)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        n_trial = len(task.config_space)
        tuner_obj.tune(
            n_trial=n_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(n_trial, prefix=prefix),
                autotvm.callback.log_to_file(log_filename),
            ],
        )


# Use graph tuner to achieve graph level optimal schedules
# Set use_DP=False if it takes too long to finish.
def tune_graph(
    graph,
    records, 
    opt_sch_file, 
    target,
    input_name="input.1", 
    dshape=(1, 3, 224, 224),
    use_DP=True
):
    target_op = [
        relay.op.get("nn.conv2d"),
    ]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {input_name: dshape}, records, target_op, target)
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)

def tune_and_evaluate(
    tuning_opt,
    model_path, 
    log_file, 
    graph_opt_sch_file,
    target,
    dev,
    input_name="input.1", 
    batch_size=1
    input_shape=(3, 224, 224),
    layout="NHWC", 
    dtype="float32", 
    use_sparse=False
):
    data_shape = (batch_size,)+input_shape
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params = get_tvm_module_N_params(model_path, input_name, batch_size, input_shape, layout, dtype, use_sparse)
    
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"),)
    )

    # run tuning tasks
    tune_kernels(tasks, **tuning_opt)
    tune_graph(mod["main"], data_shape, log_file, graph_opt_sch_file)

    # compile kernels with graph-level best records
    with autotvm.apply_graph_best(graph_opt_sch_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)

        # upload parameters to device
        data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
        module = runtime.GraphModule(lib["default"](dev))
        module.set_input(input_name, data_tvm)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", dev, number=100, repeat=3)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )
    
    return mod, params

def make_parser():
    parser = ArgumentParser(
        description=f"usage ./{__file__} ")    
    
if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    tuning_option = {
        "log_filename": cfg.log_file,
        "tuner": "random",
        "early_stopping": None,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.LocalRunner(
                number=1, repeat=10, min_repeat_ms=0, enable_cpu_cache_flush=True
            ),
        ),
    }
    mod, params = tune_and_evaluate(
        tuning_option, 
        model_path=cfg.model_path, 
        log_file=cfg.log_file, 
        graph_opt_sch_file=cfg.graph_opt_sch_file,
        target=cfg.target,
        dev=cfg.dev,
        input_name=cfg.input_name, 
        batch_size=cfg.batch_size,
        input_shape=cfg.input_shape,
        layout=cfg.layout, 
        dtype=cfg.dtype,
        use_sparse=cfg.use_sparse
        )



