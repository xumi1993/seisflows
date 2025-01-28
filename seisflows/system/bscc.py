#!/usr/bin/env python3
"""
BSCC is a system interface for the Beijing Supercomputing Center (BSCC) in China.

Information on BSCC can be found here:
https://kbs.blsc.cn
"""

import os
from datetime import timedelta

from seisflows.system.slurm import Slurm


class Bscc(Slurm):
    """
    System BSCC
    ------------
    Beijing Supercomputing Center (BSCC) in China, SLURM based system

    Parameters
    ----------
    :type partition: str
    :param partition: BSCC has various partitions which each have their
        own number of cores per compute node. Available are: gpu_4090, v6_384,
        amd_a8_384, amd_a8_768. If your partition is not given in these options,
        you can add it to the `_partitions` dictionary in the `__init__` method.
    :type submit_to: str
    :param submit_to: (Optional) partition to submit the main/master job which 
        is a serial Python task that controls the workflow. If not
        given, defaults to `partition`.

    Paths
    -----

    ***
    """
    __doc__ = Slurm.__doc__ + __doc__


    def __init__(self, mpiexec="mpiexec", partition=None, 
                 submit_to=None, ngpus=None, **kwargs):
        """BSCC init"""
        super().__init__(**kwargs)

        self.mpiexec = mpiexec
        self.partition = partition
        self.submit_to = submit_to or self.partition
        if ngpus and isinstance(ngpus, int):
            self.slurm_args = self.slurm_args or ""
            self.slurm_args += f"--gpus={ngpus}"
        self._partitions = {"gpu_4090": 6, "v6_384": 96, "amd_a8_384": 128,
                            "amd_a8_768": 128,
                            }

    @property
    def submit_call_header(self):
        """
        The submit call defines the SBATCH header which is used to submit a
        workflow task list to the system. It is usually dictated by the
        system's job scheduler. This is the header for BSCC.
        """
        _call = " ".join([
            f"sbatch",
            f"--job-name={self.title}",
            f"--output={self.path.output_log}",
            f"--error={self.path.output_log}",
            f"--ntasks=1",
            f"--partition={self.submit_to}",
            f"--time={self.walltime}"
        ])
        return _call
    
    def run_call(self, executable="", single=True, array=None, tasktime=None):
        """
        The run call defines the SBATCH header which is used to run tasks during
        an executing workflow. Like the submit call its arguments are dictated
        by the given system. Run calls are modified and called by the `run`
        function
        """
        array = array or self.task_ids(single=single)  # get job array str
        tasktime = tasktime or self.tasktime
        header = [
            f"sbatch",
            f"{self.slurm_args or ''}",
            f"--job-name={self.title}",
            f"--ntasks={self.nproc:d}",
            f"--partition={self.partition}",
            f"--time={tasktime}",
            f"--array={array}",
            f"--output={os.path.join(self.path.log_files, '%A_%a')}",
            f"--parsable",
            f"{executable}"
        ]
        _call = " ".join(header)
        return _call

