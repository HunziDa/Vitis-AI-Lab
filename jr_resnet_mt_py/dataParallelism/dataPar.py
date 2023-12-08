from ctypes import *
from typing import List
import cv2
import numpy as np
import xir
import vart
import os
import math
import threading
import time
import sys


input_tensor_size = 1000
output_tensor_size = 10

def runModelPar(runner: "Runner", runner1: "Runner", runner2: "Runner", inputs):
    
    """get tensor"""
    inputTensors = runner.get_input_tensors()
    outputTensors = runner.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    assert inputs[0].shape == input_ndim
    batch_size = len(inputs)
    output_ndim = tuple(outputTensors[0].dims)
    outputs = [np.empty(output_ndim, dtype=np.int8, order="C") for i in range(batch_size)]
    for i in range(batch_size):
        outputs[i].flags.writeable = True
    assert outputs[0].shape == output_ndim


    """prepare batch input/output """
    inputData = [[np.empty(input_ndim, dtype=np.int8, order="C")] for i in range(3)]
    outputData = [[np.empty(output_ndim, dtype=np.int8, order="C")] for i in range(3)]
    for i in range(3):
        inputData[i][0].flags.writeable = True
        outputData[i][0].flags.writeable = True
    for i in range(batch_size%3):
        inputData[0][0][:,:] = inputs[batch_size-1-i][:,:]
        job_id = runner.execute_async(inputData[0],outputData[0])
        runner.wait(job_id)
        outputs[batch_size-1-i][:,:] = outputData[0][0][:,:]
    
    for i in range(0,batch_size//3):
        inputData[0][0][:,:] = inputs[i*3][:,:]
        inputData[1][0][:,:] = inputs[i*3+1][:,:]
        inputData[2][0][:,:] = inputs[i*3+2][:,:]
        job_id = runner.execute_async(inputData[0], outputData[0])
        job_id1 = runner1.execute_async(inputData[1], outputData[1])
        job_id2 = runner2.execute_async(inputData[2], outputData[2])
        runner.wait(job_id)
        runner1.wait(job_id1)
        runner2.wait(job_id2)
        outputs[i*3][:,:] = outputData[0][0][:,:]
        outputs[i*3+1][:,:] = outputData[1][0][:,:]
        outputs[i*3+2][:,:] = outputData[2][0][:,:]
    
    return outputs



"""
 obtain dpu subgrah
"""
def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph
            is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]



def main(task_num,batch_size):
    
    #generate dataset
    data = [np.random.randint(128, size=(batch_size, input_tensor_size), dtype=np.int8) for i in range(task_num // batch_size)]

    # sub-models
    model_num = 3
    xmodel_file = "./vck190_baseline_int.xmodel"
    gemm = xir.Graph.deserialize(xmodel_file)
    subgraphs = get_child_subgraph_dpu(gemm)
    assert len(subgraphs) == 1  # only one subgraph
    outputs = []
    # print("Data format: ", len(data)," ", data[0].shape[0]," ",data[0].shape[1])

    all_dpu_runners = []
    for i in range(int(model_num)):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))
    
    time_start = time.time()
    for i in range(task_num // batch_size):
        out = runModelPar(all_dpu_runners[0], all_dpu_runners[1], all_dpu_runners[2], data)
        outputs.append(out)

    del all_dpu_runners

    time_end = time.time()
    
    # print(len(outputs[0]), outputs[0][0].shape)
    timetotal = time_end - time_start
    fps = float(task_num / timetotal)
    print(
        "FPS=%.2f, total frames = %.2f , time=%.6f seconds"
        % (fps, task_num, timetotal)
    )


if __name__ == "__main__":
    task_num = 0
    batch_size = 0
    if len(sys.argv) != 3:
        task_num = 6 * 64    
        batch_size = 6
    else:
        task_num = int(sys.argv[1])
        batch_size = int(sys.argv[2])
    main(task_num,batch_size)
