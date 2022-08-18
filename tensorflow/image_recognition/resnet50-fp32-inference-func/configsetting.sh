# python benchmarks/launch_benchmark.py --in-graph=/home/resnet50-fp32-inference/resnet50_fp32_pretrained_model.pb --model-name=resnet50 --framework=tensorflow --precision=fp32 --mode=inference --batch-size=1 --benchmark-only -socket-id=0 | tee ./log.txt
# # need tools
# /home/yangge/faas-workloads/tensorflow/image_recognition/resnet50-fp32-inference/benchmarks/common/base_model_init.py
set_num_inter_intra_threads()
num_inter_threads=1
num_intra_threads = 36 # physical cores
"OMP_NUM_THREADS": 36 # physical cores
#physical cores per socket: lscpu | grep "Core(s) per socket" | cut -d':' -f2 | xargs
#all physical cores: lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l
env_var_dict = {
    "ACCURACY_ONLY": args.accuracy_only,
    "BACKBONE_MODEL_DIRECTORY_VOL": args.backbone_model,
    "BATCH_SIZE": args.batch_size,
    "BENCHMARK_ONLY": args.benchmark_only,
    "BENCHMARK_SCRIPTS": benchmark_scripts,
    "CHECKPOINT_DIRECTORY_VOL": args.checkpoint,
    "DATASET_LOCATION_VOL": args.data_location,
    "DATA_NUM_INTER_THREADS": args.data_num_inter_threads, # defult 
    "DATA_NUM_INTRA_THREADS": args.data_num_intra_threads, # defult 
    "DISABLE_TCMALLOC": args.disable_tcmalloc,
    "DOCKER": args.docker_image or str(args.docker_image is not None),
    "DRY_RUN": str(args.dry_run) if args.dry_run is not None else "",
    "EXTERNAL_MODELS_SOURCE_DIRECTORY": args.model_source_dir,
    "FRAMEWORK": args.framework,
    "INTELAI_MODELS": intelai_models,
    "INTELAI_MODELS_COMMON": intelai_models_common,
    "MODE": args.mode,
    "MODEL_NAME": args.model_name,
    "MPI_HOSTNAMES": args.mpi_hostnames,
    "MPI_NUM_PROCESSES": args.mpi,
    "MPI_NUM_PROCESSES_PER_SOCKET": args.num_mpi,
    "NUMA_CORES_PER_INSTANCE": args.numa_cores_per_instance,
    "NOINSTALL": str(args.noinstall) if args.noinstall is not None else "True" if not args.docker_image else "False",  # noqa: E501
    "NUM_CORES": args.num_cores,
    "NUM_INTER_THREADS": args.num_inter_threads,
    "NUM_INTRA_THREADS": args.num_intra_threads,
    "NUM_TRAIN_STEPS": args.num_train_steps,
    "OUTPUT_RESULTS": args.output_results,
    "PRECISION": args.precision,
    "PYTHON_EXE": python_exe,
    "SOCKET_ID": args.socket_id,
    "TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD": args.tcmalloc_large_alloc_report_threshold,
    "TF_SERVING_VERSION": args.tf_serving_version,
    "USE_CASE": str(use_case),
    "VERBOSE": args.verbose
}