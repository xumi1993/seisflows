#!/usr/bin/env python3
"""
This class provides utilities for the Seisflows solver interactions with
Specfem3D Globe. It is built on top of the Specfem base class.

Some key differences between 3D and 3D_GLOBE implementations:

    1) SPECFEM3D_GLOBE does not require 'xgenerate_databases' which is
        required for Cartesian.
    2) Related to (1), we do NOT want to re-run the mesher, event at
        the very beginning. The Solver will simply re-load GLL model
        when run with a valid GLL model
    3) 3DGLOBE cannot toggle attenuation on/off. If it was on to create
        the model, it MUST be on for the solver (otherwise throws
        a weird 'recompile solver' error)
    4) Does not allow SU seismogram outputs, only ASCII, SAC, ASDF, 3D_Array
    5) SPECFEM3D_GLOBE ASCII synthetics have extension '.ascii', but
        the adjoint sources are NOT supposed to have this, i.e., they
        should have extension '.adj'
    6) 3DGLOBE breaks its files into 3 regions (reg1: mantle, reg2: outer core,
        reg3: inner core) which affects file bookkeeping
"""
import os
from glob import glob
from seisflows import logger
from seisflows.tools import unix
from seisflows.tools.specfem import setpar, getpar
from seisflows.solver.specfem import Specfem


class Specfem3DGlobe(Specfem):
    """
    Solver SPECFEM3D_GLOBE
    ----------------------
    SPECFEM3D_Globe-specific alterations to the base SPECFEM module

    Parameters
    ----------
    :type source_prefix: str
    :param source_prefix: Prefix of source files in path SPECFEM_DATA. Must be
        in ['CMTSOLUTION', 'FORCESOLUTION']. Defaults to 'CMTSOLUTION'
    :type export_vtk: bool
    :param export_vtk: anytime a model, kernel or gradient is considered,
        generate a VTK file and store it in the scratch/ directory for the User
        to visualize at their leisure.
    :type prune_scratch: bool
    :param prune_scratch: prune/remove database files as soon as they are used,
        to keep overall filesystem burden down
        - removes *.vt? files after they're generated by a forward simulation
        - removes proc*_absorb_field.bin and proc*_save_forward_array.bin
            files after adjoint simulations
    :type regions: str
    :param regions: which regions of the chunk  to consider in your 'model'. 
        Valid regions are 1, 2 and 3. If you want all regions, set as '123'. 
        If you only want region 1, set as  '1', etc. Order insensitive. 
    :type smooth_type: str
    :param smooth_type: choose how smoothing is performed for gradients.
        these are tied to the internal smoothing functions available.
        - 'gaussian': convolve with a 3D gaussian, slow and computationally
            intensive, but default and matches 2D and 3D_Cartesian smoothing
        - 'laplacian' (default): average points around vertex to smooth. 
            faster and preferred method for GLOBE code

    Paths
    -----
    ***
    """
    __doc__ = Specfem.__doc__ + __doc__

    def __init__(self, source_prefix="CMTSOLUTION", export_vtk=True,
                 prune_scratch=True, regions="123", smooth_type="laplacian",
                 **kwargs):
        """Instantiate a Specfem3D_Globe solver interface"""
        super().__init__(source_prefix=source_prefix, **kwargs)

        self.smooth_type = smooth_type
        self.prune_scratch = prune_scratch
        self.export_vtk = export_vtk

        # These two variables are the same but we have a public version so it 
        # will show up in the parameter file (for 3D_GLOBE only), and a private
        # one so that the other SPECFEM versions can set it as None and use it
        self.regions = str(regions)
        self._regions = sorted(self.regions) 

        # Define parameters based on material type
        if self.materials.upper() == "ACOUSTIC":
            self._parameters += ["vp"]
        elif self.materials.upper() in ["ELASTIC", "ISOTROPIC"]:
            self._parameters += ["vp", "vs"]
        elif self.materials.upper() == "ANISOTROPIC":
            self._parameters += ["vpv", "vph", "vsv", "vsh", "eta"]

        # Append regions to to the parameters, e.g., 'reg1_vpv'
        overwrite_parameters = []
        for reg in self._regions:
            overwrite_parameters.extend([f"reg{reg}_{_}" for _ in 
                                        self._parameters])
        self._parameters = sorted(overwrite_parameters)

        # Overwriting the base class parameters
        self._syn_available_data_formats = ["ASCII"]
        self._acceptable_source_prefixes = ["CMTSOLUTION", "FORCESOLUTION"]
        self._acceptable_smooth_types = ["laplacian", "gaussian"]
        self._required_binaries = ["xspecfem3D", "xmeshfem3D", "xcombine_sem"]
        if smooth_type == "laplacian":
            self._required_binaries.append("xsmooth_laplacian_sem")
        else:
            self._required_binaries.append("xsmooth_sem")

        # Internally used parameters set by functions within class
        self._model_databases = None
        self._kernel_databases = None
        self.path._vtk_files = os.path.join(self.path.scratch, "vtk_files")

    def check(self):
        """
        Checks parameter validity for SPECFEM3D_GLOBE parameters
        """
        super().check()
        
        assert(self.smooth_type in self._acceptable_smooth_types), \
            f"`smooth_type` must be in {self._acceptable_smooth_types}" 

        # Check that regions are some combination of 1, 2 and/or 3
        for r in self._regions:
            assert(int(r) in [1, 2, 3]), (
                f"`regions` must be some integer combination 1, 2 and/or 3"
                )

    def data_wildcard(self, comp="?"):
        """
        Returns a wildcard identifier for synthetic data
        Currently only support for ASCII seismograms

        :rtype: str
        :return: wildcard identifier for channels
        """
        if self.syn_data_format.upper() == "ASCII":
            return f"*.?X{comp}.sem.ascii"

    @property
    def kernel_databases(self):
        """
        The location of kernel output files, for 3DGLOBE usually DATABASES_MPI. 
        This can be determined by 'LOCAL_PATH' in the Par_file. Points to
        `model_databases` as it is assumed they are the same path
        """
        if self._kernel_databases is None:
            self._kernel_databases = self.model_databases

        return self._kernel_databases

    @property
    def model_databases(self):
        """
        The location of model output files, for 3DGLOBE usually DATABASES_MPI. 
        This can be determined by 'LOCAL_PATH' in the Par_file and must be 
        defined relative to the main directory
        """
        if self._model_databases is None:
            self._model_databases = getpar(
                key="LOCAL_PATH", file=os.path.join(self.path.specfem_data,
                                                    "Par_file"))[1]
        return os.path.basename(self._model_databases)

    def forward_simulation(self, executables=None, save_traces=False,
                           export_traces=False, **kwargs):
        """
        Calls SPECFEM3D_GLOBE forward solver, exports solver outputs to traces.



        :type executables: list or None                                          
        :param executables: list of SPECFEM executables to run, in order, to     
            complete a forward simulation. This can be left None in most cases,  
            which will select default values based on the specific solver        
            being called (2D/3D/3D_GLOBE). It is made an optional parameter      
            to keep the function more general for inheritance purposes.          
        :type save_traces: str                                                   
        :param save_traces: move files from their native SPECFEM output location 
            to another directory. This is used to move output waveforms to       
            'traces/obs' or 'traces/syn' so that SeisFlows knows where to look   
            for them, and so that SPECFEM doesn't overwrite existing files       
            during subsequent forward simulations                                
        :type export_traces: str                                                 
        :param export_traces: export traces from the scratch directory to a more 
            permanent storage location. i.e., copy files from their original     
            location 
        """
        if executables is None:
            executables = ["bin/xspecfem3D"]

        super().forward_simulation(executables=executables, 
                                   save_traces=save_traces, 
                                   export_traces=export_traces, 
                                   **kwargs)

        if self.prune_scratch:
            logger.debug("removing '*.vt?' files from database directory")
            unix.rm(glob(os.path.join(self.model_databases, "proc*_*.vt?")))

    def adjoint_simulation(self, executables=None, save_kernels=False,
                           export_kernels=False):
        """
        Supers SPECFEM for adjoint solver and removes GLOBE-specific fwd files
        Also deals with anisotropic kernels (or lack thereof)

        :type executables: list or None                                          
        :param executables: list of SPECFEM executables to run, in order, to     
            complete an adjoint simulation. This can be left None in most cases, 
            which will select default values based on the specific solver        
            being called (2D/3D/3D_GLOBE). It is made an optional parameter      
            to keep the function more general for inheritance purposes.          
        :type save_kernels: str                                                  
        :param save_kernels: move the kernels from their native SPECFEM output   
            location to another path. This is used to move kernels to another    
            SeisFlows scratch directory so that they are discoverable by         
            other modules. The typical location they are moved to is             
            path_eval_grad                                                       
        :type export_kernels: str                                                
        :param export_kernels: export/copy/save kernels from the scratch         
            directory to a more permanent storage location. i.e., copy files     
            from their original location. Note that kernel file sizes are LARGE, 
            so exporting kernels can lead to massive storage requirements.
        """
        if executables is None:
            executables = ["bin/xspecfem3D"]

        # Make sure we have a STATIONS_ADJOINT file. Simply copy STATIONS file
        dst = os.path.join(self.cwd, "DATA", "STATIONS_ADJOINT")
        if not os.path.exists(dst):
            src = os.path.join(self.cwd, "DATA", "STATIONS")
            unix.cp(src, dst)

        # Control the kernel quantities generated by SPECFEM
        if self.materials.upper() in ["ACOUSTIC", "ELASTIC", "ISOTROPIC"]:
            anisotropic_kl = ".false."
            save_transverse_kl_only = ".false."
        elif self.materials.upper() == "ANISOTROPIC":
            anisotropic_kl = ".true."
            save_transverse_kl_only = ".true."
        elif self.materials.upper() == "FULLY_ANISOTROPIC":
            # Work in progress, setting up for full 21 parameter anisotropy
            raise NotImplementedError("Full anisotropy is not yet implemented "
                                      "in SeisFlows")
            anisotropic_kl = ".true."
            save_transverse_kl_only = ".false."

        unix.cd(self.cwd)
        setpar(key="ANISOTROPIC_KL", val=anisotropic_kl, file="DATA/Par_file")
        setpar(key="SAVE_TRANSVERSE_KL_ONLY", val=save_transverse_kl_only, 
               file="DATA/Par_file")
        
        # SPECFEM3D class takes care of attenuation and STATIONS_ADJOINT file
        super().adjoint_simulation(executables=executables,                      
                                   save_kernels=save_kernels,                    
                                   export_kernels=export_kernels)

        # Working around fact that `absorb_buffer` files have diff naming w.r.t
        # SPECFEM3D. Will also remove `save_forward_arrays`
        if self.prune_scratch:                                                   
            for glob_key in ["proc??????_reg?_absorb_buffer.bin"]: 
                logger.debug(f"removing '{glob_key}' files from database "       
                             f"directory")                                       
                unix.rm(glob(os.path.join(self.model_databases, glob_key)))

    def combine(self, input_path, output_path, parameters=None):                 
        """
        Overwrite of xcombine_sem with an additional file check as 
        SPECFEM3D_GLOBE requires file 'mesh_parameters.bin'
                                                                                 
        :type input_path: str                                                    
        :param input_path: path to data                                          
        :type output_path: strs                                                  
        :param output_path: path to export the outputs of xcombine_sem           
        :type parameters: list                                                   
        :param parameters: optional list of parameters,                          
            defaults to `self._parameters`                                       
        """
        if parameters is None:
            parameters = self._parameters

        # Switch to relative pathing
        unix.cd(self.cwd)

        # Copy the 'mesh_parameters.bin' from LOCAL_PATH. Assumed to be the 
        # same for all tasks
        src = os.path.join(self.model_databases, "mesh_parameters.bin")
        for name in self.source_names:
            dst = os.path.join(input_path, name, "mesh_parameters.bin")
            unix.cp(src, dst)
        
        # 3DGLOBE 'xcombine_sem' does not expect `reg?_` prefix, strip off
        stripped_parameters = list(set([_[5:] for _ in parameters]))

        super().combine(input_path=input_path, output_path=output_path,
                        parameters=stripped_parameters)

    def smooth(self, input_path, output_path, parameters=None, span_h=None,      
               span_v=None, use_gpu=False):                                      
        """
        Logic function to choose between available smoothing types for GLOBE
                                                                                 
        :type input_path: str                                                    
        :param input_path: path to data                                          
        :type output_path: str                                                   
        :param output_path: path to export the outputs of xcombine_sem           
        :type parameters: list                                                   
        :param parameters: optional list of parameters,                          
            defaults to `self._parameters`                                       
        :type span_h: float                                                      
        :param span_h: horizontal smoothing length in meters                     
        :type span_v: float                                                      
        :param span_v: vertical smoothing length in meters                       
        :type use_gpu: bool                                                      
        :param use_gpu: whether to use GPU acceleration for smoothing. Requires  
            GPU compiled binaries and GPU compute node.                          
        """
        if parameters is None:
            parameters = self._parameters

        # 3DGLOBE 'xsmooth_*sem' does not expect `reg?_` prefix, strip off
        stripped_parameters = list(set([_[5:] for _ in parameters]))

        if self.smooth_type == "gaussian":
            super().smooth(input_path=input_path, output_path=output_path, 
                           parameters=stripped_parameters, span_h=span_h, 
                           span_v=span_v, use_gpu=use_gpu)
        elif self.smooth_type == "laplacian":
            self.smooth_laplacian(
                    input_path=input_path, output_path=output_path, 
                    parameters=stripped_parameters, span_h=span_h, span_v=span_v
                    )

    def smooth_laplacian(self, input_path, output_path, parameters=None, 
                         span_h=None, span_v=None):
        """                                                                      
        Wrapper for SPECFEM binary: xsmooth_laplacian_sem

        Smooths kernels by with Laplacian smoothing which takes averages of a
        mesh corner with all it's surrounding points.

        .. note::
            Externally this smooth function behaves almost identically to
            the normal gaussian smoothing function
                                                                                 
        .. note::                                                                
            It is ASSUMED that this function is being called by                  
            system.run(single=True) so that we can use the main solver           
            directory to perform the kernel smooth task                          
                                                                                 
        :type input_path: str                                                    
        :param input_path: path to data                                          
        :type output_path: str                                                   
        :param output_path: path to export the outputs of xcombine_sem           
        :type parameters: list                                                   
        :param parameters: optional list of parameters,                          
            defaults to `self._parameters`                                       
        :type span_h: float                                                      
        :param span_h: horizontal smoothing length in km
        :type span_v: float                                                      
        :param span_v: vertical smoothing length in km
        """                                                                      
        unix.cd(self.cwd)                                                        
                                                                                 
        # Assign some default parameters from class attributes if not given      
        if parameters is None:                                                   
            parameters = self._parameters                                        
        if span_h is None:                                                       
            span_h = self.smooth_h 
        if span_v is None:                                                       
            span_v = self.smooth_v
                                                                                 
        logger.debug(f"smoothing {parameters} with laplacian, horizontal span "
                     f"{span_h}m and vertical span {span_v}m")               

        # NOTE: Converting smoothing lengths 'm' -> 'km' as laplacian smoothing
        #   function is epxecting things in 'km' while SeisFlows expects things
        #   in 'm'
        span_h *= 1E-3
        span_v *= 1E-3

        if not os.path.exists(output_path):                                      
            unix.mkdir(output_path)                                              
                                                                                 
        # Ensure trailing '/' character, required by xsmooth_sem                 
        input_path = os.path.join(input_path, "")                                
        output_path = os.path.join(output_path, "")                              

        # mpiexec ./bin/xsmooth_laplacian_sem SIGMA_H SIGMA_V name input output
        for name in parameters:                                                  
            exc = (f"bin/xsmooth_laplacian_sem {str(span_h)} {str(span_v)} "
                   f"{name}_kernel {input_path} {output_path}")
            # e.g., combine_vs.log                                               
            stdout = f"{self._exc2log(exc)}_{name}.log"                          
            self._run_binary(executable=exc, stdout=stdout)                      
                                                                                 
        # Rename output files to remove the '_smooth' suffix which SeisFlows     
        # will not recognize                                                     
        files = glob(os.path.join(output_path, "*"))                             
        unix.rename(old="_smooth", new="", names=files)
        
    def combine_vol_data_vtk(self, input_path, output_path, hi_res=False,
                             parameters=None):
        """
        Wrapper for 'xcombine_vol_data_vtk'. Combines binary files together
        to generate a single .VTK file that can be visualized by external
        software like ParaView

        .. rubric::
            xcombine_data start end quantity input_dir output_dir hi/lo-res

        .. note::
            It is ASSUMED that this function is being called by
            system.run(single=True) so that we can use the main solver
            directory to perform the kernel summation task

        :type input_path: str
        :param input_path: path to database files to be summed.
        :type output_path: strs
        :param output_path: path to export the outputs of xcombine_sem
        :type hi_res: bool
        :param hi_res: Set the high resolution flag to 1 or True, which will
            generate .vtk files with data at EACH GLL point, rather than at each
            nodal vertex. These files are LARGE, and we discourage using
            `hi_res`==True unless you know you want these files.
        :type parameters: list
        :param parameters: optional list of parameters,
            defaults to `self._parameters`
        """
        unix.cd(self.cwd)

        if parameters is None:
            parameters = self._parameters

        if not os.path.exists(output_path):
            unix.mkdir(output_path)

        # Call on xcombine_sem to combine kernels into a single file
        for name in parameters:
            # e.g.:  bin/xcombine_vol_data_vtk 0 3 alpha_kernel in/ out/ 0
            exc = f"bin/xcombine_vol_data_vtk 0 {self.nproc-1} {name} " \
                  f"{input_path} {output_path} {int(hi_res)}"
            # e.g., smooth_vp.log
            stdout = f"{self._exc2log(exc)}_{name}.log"
            self._run_binary(executable=exc, stdout=stdout, with_mpi=False)
