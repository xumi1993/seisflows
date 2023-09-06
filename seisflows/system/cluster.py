#!/usr/bin/env python3
"""
The Cluster class provides the core utilities interaction with HPC systems
which must be overloaded by subclasses for specific workload managers, or
specific clusters.

The `Cluster` class acts as a base class for more specific cluster
implementations (like SLURM). However it can be used standalone. When running
jobs on the `Cluster` system, jobs will be submitted to the master system
using `subprocess.run`, mimicing how jobs would be run on a cluster but not
actually submitting to any job scheduler.
"""
import os
import sys
import subprocess
from concurrent.futures import ProcessPoolExecutor, wait
from seisflows import logger, ROOT_DIR
from seisflows.tools import msg
from seisflows.tools.unix import nproc
from seisflows.tools.config import pickle_function_list, copy_file
from seisflows.system.workstation import Workstation


class Cluster(Workstation):
    """
    Cluster System
    --------------
    Generic or common HPC/cluster interfacing commands

    Parameters
    ----------
    :type title: str
    :param title: The name used to submit jobs to the system, defaults
        to the name of the current working directory
    :type mpiexec: str
    :param mpiexec: Function used to invoke executables on the system.
        For example 'mpirun', 'mpiexec', 'srun', 'ibrun'
    :type ntask_max: int
    :param ntask_max: limit the number of concurrent tasks in a given array job.
        Note that if you are directly running the Cluster system on a 
        workstation or the login node of your cluster, try to adhere to 
        the number of cores on your system should be <= `ntask_max` * `nproc`,
        otherwise you may face memory allocation errors
    :type walltime: float
    :param walltime: maximum job time in minutes for the master SeisFlows
        job submitted to cluster. Fractions of minutes acceptable.
    :type tasktime: float
    :param tasktime: maximum job time in minutes for each job spawned by
        the SeisFlows master job during a workflow. These include, e.g.,
        running the forward solver, adjoint solver, smoother, kernel combiner.
        All spawned tasks receive the same task time. Fractions of minutes
        acceptable.
    :type environs: str
    :param environs: Optional environment variables to be provided in the
        following format VAR1=var1,VAR2=var2... Will be set using
        os.environs

    Paths
    -----
    ***
    """
    __doc__ = Workstation.__doc__ + __doc__

    # These are class paths that specify the location of run and submit scripts
    # Setting them outside init allows them to be inherited by all child classes
    # implicitely. Should not be edited unless using custom submit/run scripts
    submit_workflow = os.path.join(ROOT_DIR, "system", "runscripts", "submit")
    run_functions = os.path.join(ROOT_DIR, "system", "runscripts", "run")

    def __init__(self, title=None, mpiexec="", ntask_max=None, walltime=10,
                 tasktime=1, environs="", **kwargs):
        """Instantiate the Cluster System class"""
        super().__init__(**kwargs)

        if title is None:
            self.title = os.path.basename(os.getcwd())
        else:
            self.title = title
        self.mpiexec = mpiexec
        self.ntask_max = ntask_max or nproc() - 1  # -1 because master job
        self.walltime = walltime
        self.tasktime = tasktime
        self.environs = environs or ""

    @property
    def submit_call_header(self):
        """
        The submit call defines the SBATCH header which is used to submit a
        workflow task list to the system. It is usually dictated by the
        system's required parameters, such as account names and partitions.
        Submit calls are modified and called by the `submit` function.

        .. note::
            Generalized `cluster` returns empty string but child system
            classes will need to overwrite the submit call.

        :rtype: str
        :return: the system-dependent portion of a submit call
        """
        return ""

    @property
    def run_call_header(self):
        """
        The run call defines the SBATCH header which is used to run tasks during
        an executing workflow. Like the submit call its arguments are dictated
        by the given system. Run calls are modified and called by the `run`
        function

        .. note::
            Generalized `cluster` returns empty string but child system
            classes will need to overwrite the submit call.

        :rtype: str
        :return: the system-dependent portion of a run call
        """
        return ""


    def submit(self, workdir=None, parameter_file="parameters.yaml", 
               login=False):
        """
        Submits the main workflow job as a separate job submitted directly to
        the system that is running the master job

        :type workdir: str
        :param workdir: path to the current working directory
        :type parameter_file: str
        :param parameter_file: paramter file file name used to instantiate
            the SeisFlows package
        :type login: bool
        :param login: (used for overriding system modules) submits the main 
            workflow job directly to the login node rather than as a separate 
            process on a compute node. Avoids queue times and walltimes but may 
            be discouraged by sys admins as some  array processing will take 
            place on the shared login node. 
        """
        # Copy log files if present to avoid overwriting
        for src in [self.path.output_log, self.path.par_file]:
            if os.path.exists(src) and os.path.exists(self.path.log_files):
                copy_file(src, copy_to=self.path.log_files)

        # Determine where submit call will be sent (login or compute node)
        if login:
            header = ""
            logger.info("submitting master job on login node")
        else:
            header = self.submit_call_header

        # e.g., submit -w ./ -p parameters.yaml
        submit_call = " ".join([
            f"{header}",
            f"{self.submit_workflow}",
            f"--workdir {workdir}",
            f"--parameter_file {parameter_file}",
        ])

        logger.debug(submit_call)
        try:
            subprocess.run(submit_call, shell=True)
        except subprocess.CalledProcessError as e:
            logger.critical(f"SeisFlows master job has failed with: {e}")
            sys.exit(-1)

    def run(self, funcs, single=False, **kwargs):
        """
        Runs tasks multiple times in parallel by submitting NTASK new jobs to
        system. The list of functions and its kwargs are saved as pickles files,
        and then re-loaded by each submitted process with specific environment
        variables. Each spawned process will run the list of functions.

        .. warning::

            Logging parameters `verbose` and `log_level` are passed to each
            compute job through the pickled kwargs file. This assumes that NONE
            of the functions that are being passed through also have these 
            variable names as arguments, otherwise there will be conflicting
            arguments. Probably this won't be the case because these variable
            names are generally reserved for logging purposes.

        :type funcs: list of methods
        :param funcs: a list of functions that should be run in order. All
            kwargs passed to run() will be passed into the functions.
        :type single: bool
        :param single: run a single-process, non-parallel task, such as
            smoothing the gradient, which only needs to be run by once.
            This will change how the job array and the number of tasks is
            defined, such that the job is submitted as a single-core job to
            the system.
        :type run_call: str
        :param run_call: the call used to submit the run script. If None,
            attempts default run call which should be suited for the given
            system. Can be overwritten by child classes to involve other
            arguments
        """
        # Single tasks only need to be run one time, as `TASK_ID` === 0
        task_ids = self.task_ids(single)
        funcs_fid, kwargs_fid = pickle_function_list(functions=funcs,
                                                     path=self.path.scratch,
                                                     verbose=self.verbose,
                                                     level=self.log_level,
                                                     **kwargs)
        logger.info(f"running functions {[_.__name__ for _ in funcs]} on "
                    f"system {len(task_ids)} times")

        # Create the run call which will simply call an external Python script
        # e.g., run --funcs func.p --kwargs kwargs.p --environment ...
        run_call = " ".join([
            f"{self.run_call_header}",
            f"{self.run_functions}",
            f"--funcs {funcs_fid}",
            f"--kwargs {kwargs_fid}",
            f"--environment SEISFLOWS_TASKID={{task_id}},{self.environs}"
        ])
        logger.debug(run_call)

        # Don't need to spin up concurrent.futures for a single run so we just
        # call a function instead
        if single:
            self._run_task(run_call=run_call, task_id=0)
        # Run tasks in parallel and wait for all of them to finish
        else:
            with ProcessPoolExecutor(max_workers=self.ntask_max) as executor:
                futures = [executor.submit(self._run_task, run_call, task_id)
                           for task_id in task_ids]

            # Mimic a cluster job timeout by limiting task to `tasktime`
            wait(futures, timeout=self.tasktime, return_when="FIRST_EXCEPTION")

            # Iterate through the Futures' results because if one of them
            # raised an exception, it will break the main process here
            # https://stackoverflow.com/questions/33448329/
            #   how-to-detect-exceptions-in-concurrent-futures-in-python3
            failed_jobs = []
            for future in futures:
                job_id, return_code = future.result()
                if return_code != 0:
                    failed_jobs.append(job_id)

            if failed_jobs:
                # Non-zero return codes means underlying job failures
                logger.critical(
                    msg.cli(f"The system attempt to run the given `run call` "
                            f"for the following `task ids` has returned a "
                            f"non-zero exit code suggesting external job "
                            f"failure(s). Please check the "
                            f"corresponding log files in `logs/` for more "
                            f"detailed error message(s)",
                            header="system run error", border="=",
                            items=[f"RUN CALL: {run_call}",
                                   f"TASK IDS: {failed_jobs}"]
                            )
                )
                sys.exit(-1)

    def _run_task(self, run_call, task_id):
        """
        Convenience function to run a single Python job with subprocess.run
        with some error catching and redirect of stdout to a log file.

        :type run_call: str
        :param run_call: python call to run a task involving loading the
            pickled function list and its kwargs, and then running them
        :type task_id: int
        :param task_id: given task id, used for log messages and to format
            the run call
        """
        logger.debug(f"running task id {task_id}")
        try:
            f = open(self._get_log_file(task_id), "w")
            result = subprocess.run(run_call.format(task_id=task_id),
                                    shell=True, stdout=f)
        except subprocess.CalledProcessError as e:
            logger.critical(f"run task_id {task_id} has failed with error "
                            f"message {e}")
            sys.exit(-1)
        finally:
            f.close()

        return task_id, result.returncode
