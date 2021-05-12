import os
import argparse
import cdsw
import ray
import time


# TODO: using this function + command line args does not work!
def initialize_ray_cluster(args):
    RAY_DASHBOARD_PORT = int(os.getenv("CDSW_READONLY_PORT"))

    ray_head = ray.init() #dashboard_port=RAY_DASHBOARD_PORT
    #r!time.sleep(5)
    ray_nodes = cdsw.launch_workers(
        n=args.workers,
        cpu=args.cpus,
        memory=args.memory,
        kernel="python3",
        code=f"!ray start --num-cpus={args.cpus} --address={ray_head['redis_address']}; while true; do sleep 10; done",
    )
    print(
        f"""http://read-only-{os.getenv('CDSW_MASTER_ID')}.{os.getenv("CDSW_DOMAIN")}"""
    )
    # Set environment variable so other scripts can access the head address
    os.environ["RAY_CLUSTER_ADDRESS"] = ray_head["redis_address"]
    return ray_nodes


def teardown_ray_cluster(ray_nodes):
    ray.shutdown()
    cdsw.stop_workers(*[worker["id"] for worker in ray_nodes])


# TODO: this doesn't work right now
parser = argparse.ArgumentParser()
parser.add_argument(
    "--workers",
    default=2,
    type=int,
    help="number of ray workers to initialize (total number of nodes will be workers + 1)",
)
parser.add_argument(
    "--cpus", default=1, type=int, help="number of CPUs to allocate per worker"
)
parser.add_argument(
    "--memory", default=2, type=int, help="amount of memory to allocate to each worker"
)
args, _ = parser.parse_known_args()

ray_nodes = initialize_ray_cluster(args)
