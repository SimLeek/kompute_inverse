"""Example for finding the inverse of multiple matrices using Kompute."""

import time
import kp
import numpy as np
import subprocess
import os
import sys


# todo: currently, doesn't handle 0s in diagonal.
#  so:
#  1. replace the identity mat with a temp mat
#  2. set each entry to s(x)=abs((v-1)/((v-e)(v-1/e))), where v is the i,j entry value and e is machine eps
#  3. set each entry to s(x)=1/(s(x_-1)+1)+1/(s(x_-2)+1)+...+1/(s(x_+1)+1)+1/(s(x_+2)+...
#   3a. to optimize this for large matrices, use a reduce op
#  given step 4, step 3 is unnecessary
#  4. perform "merge sort" on the rows, where each row value is equal to the value of its position index within its row
#   4a., if the overall value of two items is lower swapped, swap them
#   4b. use bitonic merge sort. Add l<max to if l>i. for loops on k and j. size < 1024.
#   4c. only change is arr[i]>arr[l] becomes tmp[i*n+arr[i]]+tmp[l*n+arr[l]]<tmp[i*n+arr[l]]+tmp[l*n+arr[i]], and arr starts as 0,1,2,3,...
#       x*n+y format
#   4d. shared mem isn't too useful since we have to get from the tmp array to tell the value
#  5. swap all the rows given the sorted indices, and set up the tmp matrix as a swapped identity matrix


def get_shader(filename):
    """Get a spir-v shader if it exists, otherwise generate it."""

    # Check if the filename ends with .glsl or .comp
    if filename.endswith(".glsl") or filename.endswith(".comp"):
        # Check if a compiled .spv version of the file exists
        spv_filename = filename[:-5] + ".spv"
        if os.path.exists(spv_filename):
            # Read the compiled .spv file
            with open(spv_filename, "rb") as f:
                shader = f.read()
        else:
            # Compile the GLSL file using glslc
            try:
                result = subprocess.run(
                    ["glslc", filename, "-o", spv_filename],
                    check=True,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                # Print the full output from the glslc command
                print("glslc command failed with output:")
                print(e.stdout)
                print(e.stderr, file=sys.stderr)
                raise e

            # Wait for the glslc command to finish and check for errors
            with open(spv_filename, "rb") as f:
                shader = f.read()
    else:
        raise ValueError(
            "Invalid file extension. Filename must end with .glsl or .comp"
        )

    return shader


def mat_inv():
    """Run an example multi-matrix inverse compute operation."""

    # with capsys.disabled():
    manager = kp.Manager()
    # needed for optimization
    max_workgroup_invocations = manager.get_device_properties()[
        "max_work_group_invocations"
    ]
    maxComputeSharedMemorySize = 49152

    matrix_10_10 = (
        np.asarray(
            [
                [2, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                [2, 4, 3, 4, 5, 6, 7, 8, 9, 10],
                [2, 4, 6, 4, 5, 6, 7, 8, 9, 10],
                [2, 4, 6, 8, 5, 6, 7, 8, 9, 10],
                [2, 4, 6, 8, 10, 6, 7, 8, 9, 10],
                [2, 4, 6, 8, 10, 12, 7, 8, 9, 10],
                [2, 4, 6, 8, 10, 12, 14, 8, 9, 10],
                [2, 4, 6, 8, 10, 12, 14, 16, 9, 10],
                [2, 4, 6, 8, 10, 12, 14, 16, 18, 10],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            ]
        )
        / 18.0
    )

    identity_10 = np.eye(10)

    matrix_11_11 = (
        np.asarray(
            [
                [2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                [2, 4, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                [2, 4, 6, 4, 5, 6, 7, 8, 9, 10, 11],
                [2, 4, 6, 8, 5, 6, 7, 8, 9, 10, 11],
                [2, 4, 6, 8, 10, 6, 7, 8, 9, 10, 11],
                [2, 4, 6, 8, 10, 12, 7, 8, 9, 10, 11],
                [2, 4, 6, 8, 10, 12, 14, 8, 9, 10, 11],
                [2, 4, 6, 8, 10, 12, 14, 16, 9, 10, 11],
                [2, 4, 6, 8, 10, 12, 14, 16, 18, 10, 11],
                [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 11],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            ]
        )
        / 20.0
    )

    identity_11 = np.eye(11)

    matrix_3_3 = np.asarray([[0, 1, 0], [2, 0, 0], [0, 0, 3]])

    identity_3 = np.eye(3)

    # source: https://www.wolframalpha.com/input?i2d=true&i=Divide%5B%7B%7B2%2C2%2C3%2C4%2C5%2C6%2C7%2C8%2C9%2C10%7D%2C%7B2%2C4%2C3%2C4%2C5%2C6%2C7%2C8%2C9%2C10%7D%2C%7B2%2C4%2C6%2C4%2C5%2C6%2C7%2C8%2C9%2C10%7D%2C%7B2%2C4%2C6%2C8%2C5%2C6%2C7%2C8%2C9%2C10%7D%2C%7B2%2C4%2C6%2C8%2C10%2C6%2C7%2C8%2C9%2C10%7D%2C%7B2%2C4%2C6%2C8%2C10%2C12%2C7%2C8%2C9%2C10%7D%2C%7B2%2C4%2C6%2C8%2C10%2C12%2C14%2C8%2C9%2C10%7D%2C%7B2%2C4%2C6%2C8%2C10%2C12%2C14%2C16%2C9%2C10%7D%2C%7B2%2C4%2C6%2C8%2C10%2C12%2C14%2C16%2C18%2C10%7D%2C%7B1%2C2%2C3%2C4%2C5%2C6%2C7%2C8%2C9%2C10%7D%7D%2C18%5D
    matrix_10_10_inv = (
        np.asarray(
            [
                [2520, 0, 0, 0, 0, 0, 0, 0, 0, -2520],
                [-1260, 1260, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, -840, 840, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, -630, 630, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, -504, 504, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, -420, 420, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, -360, 360, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, -315, 315, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, -280, 280, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, -252, 504],
            ]
        )
        / 140.0
    )

    # source: https://www.wolframalpha.com/input?i2d=true&i=Divide%5B%7B%7B2%2C2%2C3%2C4%2C5%2C6%2C7%2C8%2C9%2C10%2C11%7D%2C%7B2%2C4%2C3%2C4%2C5%2C6%2C7%2C8%2C9%2C10%2C11%7D%2C%7B2%2C4%2C6%2C4%2C5%2C6%2C7%2C8%2C9%2C10%2C11%7D%2C%7B2%2C4%2C6%2C8%2C5%2C6%2C7%2C8%2C9%2C10%2C11%7D%2C%7B2%2C4%2C6%2C8%2C10%2C6%2C7%2C8%2C9%2C10%2C11%7D%2C%7B2%2C4%2C6%2C8%2C10%2C12%2C7%2C8%2C9%2C10%2C11%7D%2C%7B2%2C4%2C6%2C8%2C10%2C12%2C14%2C8%2C9%2C10%2C11%7D%2C%7B2%2C4%2C6%2C8%2C10%2C12%2C14%2C16%2C9%2C10%2C11%7D%2C%7B2%2C4%2C6%2C8%2C10%2C12%2C14%2C16%2C18%2C10%2C11%7D%2C%7B2%2C4%2C6%2C8%2C10%2C12%2C14%2C16%2C18%2C20%2C11%7D%2C%7B1%2C2%2C3%2C4%2C5%2C6%2C7%2C8%2C9%2C10%2C11%7D%7D%2C20%5D
    matrix_11_11_inv = (
        np.asarray(
            [
                [27720, 0, 0, 0, 0, 0, 0, 0, 0, 0, -27720],
                [-13860, 13860, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, -9240, 9240, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, -6930, 6930, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, -5544, 5544, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, -4620, 4620, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, -3960, 3960, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, -3465, 3465, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, -3080, 3080, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, -2772, 2772, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, -2520, 5040],
            ]
        )
        / 1386.0
    )

    # source: https://www.wolframalpha.com/input?i=invert%3A+%7B%7B0%2C1%2C0%7D%2C%7B2%2C0%2C0%7D%2C%7B0%2C0%2C3%7D%7D
    matrix_3_3_inv = np.asarray([
        [0,3,0],
        [6,0,0],
        [0,0,2]
    ]
    )/6.0

    shad_norm = get_shader("shaders/normalize.comp")
    shad_gauss = get_shader("shaders/gauss_jordan.comp")
    shad_iter = get_shader("shaders/iter.comp")

    buf_A = manager.tensor(
        np.concatenate(
            [matrix_10_10.flatten(), matrix_11_11.flatten(), matrix_3_3.flatten()]
        )
    )
    buf_I = manager.tensor(
        np.concatenate(
            [identity_10.flatten(), identity_11.flatten(), identity_3.flatten()]
        )
    )
    # buf_A = manager.tensor(np.concatenate([identity_10.flatten(), identity_11.flatten()]))
    # sending context: 2 matricies, first is 10x10 and starts at index 0, second is 11x11 and starts at index 100
    buf_D = manager.tensor(
        np.asarray([2, 10, 0, 11, 100, 3, 221], dtype=np.uint32).view(np.float32)
    )
    buf_i = manager.tensor(np.asarray([0], dtype=np.uint32).view(np.float32))

    manager.sequence().eval(kp.OpTensorSyncDevice([buf_A, buf_I, buf_D, buf_i]))

    # FORWARD PASS
    algorithm_norm = manager.algorithm(
        [buf_A, buf_I, buf_D, buf_i],
        spirv=shad_norm,
        workgroup=[int(np.ceil(buf_A.size() / max_workgroup_invocations)), 0, 0],
        spec_consts=np.asarray([max_workgroup_invocations], dtype=np.uint32).view(
            np.float32
        ),
    )
    algorithm_gauss = manager.algorithm(
        [buf_A, buf_I, buf_D, buf_i],
        spirv=shad_gauss,
        workgroup=[int(np.ceil(buf_A.size() / max_workgroup_invocations)), 0, 0],
        spec_consts=np.asarray([max_workgroup_invocations], dtype=np.uint32).view(
            np.float32
        ),
    )
    algorithm_iter = manager.algorithm(
        [buf_i],
        spirv=shad_iter,
        workgroup=[1, 0, 0],
        spec_consts=np.asarray([1], dtype=np.uint32).view(np.float32),
    )

    sequence_full = manager.sequence()
    sequence_full.record(kp.OpTensorSyncDevice([buf_A, buf_I, buf_D, buf_i]))
    for i in range(max(matrix_10_10.shape[0], matrix_11_11.shape[0])):
        sequence_full.record(kp.OpAlgoDispatch(algorithm_norm))
        sequence_full.record(kp.OpAlgoDispatch(algorithm_gauss))
        sequence_full.record(kp.OpAlgoDispatch(algorithm_iter))
    sequence_full.record(kp.OpTensorSyncLocal([buf_A, buf_I, buf_D, buf_i]))

    t1 = time.time()
    sequence_full.eval()
    t2 = time.time()
    np.set_printoptions(precision=3, linewidth=150, suppress=True)

    print(f"eval time: {(t2 - t1) * 1e3} milliseconds.")
    print(f"10x10 inverse: \n{(buf_I.data()[:100].reshape(10, 10))}.")
    #print(f"10x10 tmp: \n{(buf_A.data()[:100].reshape(10, 10))}.")
    print(f"expected 10x10 inverse: \n{matrix_10_10_inv}")
    print(f"expected-actual diff: \n{abs(matrix_10_10_inv-buf_I.data()[0:100].reshape(10, 10))}")
    print(f"total diff: {np.sum(abs(matrix_10_10_inv-buf_I.data()[0:100].reshape(10, 10)))}")
    print(f"11x11 inverse: \n{(buf_I.data()[100:221].reshape(11, 11))}.")
    #print(f"11x11 tmp: \n{(buf_A.data()[100:221].reshape(11, 11))}.")
    print(f"expected 11x11 inverse: \n{matrix_11_11_inv}")
    print(f"expected-actual diff: \n{abs(matrix_11_11_inv-buf_I.data()[100:221].reshape(11, 11))}")
    print(f"total diff: {np.sum(abs(matrix_11_11_inv-buf_I.data()[100:221].reshape(11, 11)))}")
    print(f"3x3 inverse: \n{(buf_I.data()[221:230].reshape(3, 3))}.")
    print(f"expected 3x3 inverse: \n{matrix_3_3_inv}.")
    print(f"expected-actual diff: \n{abs(matrix_3_3_inv-buf_I.data()[221:230].reshape(3, 3))}")
    print(f"total diff: {np.sum(abs(matrix_3_3_inv-buf_I.data()[221:230].reshape(3, 3)))}")
    #print(f"3x3 tmp: \n{(buf_A.data()[221:230].reshape(3, 3))}.")


if __name__ == "__main__":
    mat_inv()
