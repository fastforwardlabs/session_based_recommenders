import os
import argparse
import cdsw
import ray
import time


RAY_DASHBOARD_PORT = int(os.getenv("CDSW_READONLY_PORT"))

WORKERS = 5
CPUS = 1
MEMORY = 2

### RUN THE FOLLOWING LINES TO INITIALIZE A RAY CLUSTER IN CDSW/CML SESSION
ray_head = ray.init(dashboard_port=RAY_DASHBOARD_PORT)
ray_nodes = cdsw.launch_workers(
    n=WORKERS,
    cpu=CPUS,
    memory=MEMORY,
    kernel="python3",
    code=f"!ray start --num-cpus={CPUS} --address={ray_head['redis_address']}; while true; do sleep 10; done",
)
print(
    f"""http://read-only-{os.getenv('CDSW_MASTER_ID')}.{os.getenv("CDSW_DOMAIN")}"""
)
# Set environment variable so other scripts can access the head address
os.environ["RAY_CLUSTER_ADDRESS"] = ray_head["redis_address"]


### RUN THESE LINES TO TEAR DOWN RAY CLUSTER WHEN FINISHED
ray.shutdown()
cdsw.stop_workers(*[worker["id"] for worker in ray_nodes])



