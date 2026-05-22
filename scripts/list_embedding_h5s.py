import h5py
from pathlib import Path


def main():
    for path in sorted(Path("datasets").glob("*.h5")):
        try:
            with h5py.File(path, "r") as f:
                groups = total = flow = 0
                examples = []
                for group_name, group in f.items():
                    if not isinstance(group, h5py.Group):
                        continue
                    groups += 1
                    if len(examples) < 5:
                        examples.append(group_name)
                    trajs = [
                        key
                        for key in group.keys()
                        if "lang" not in key and not key.startswith("flow_")
                    ]
                    total += len(trajs)
                    flow += sum(
                        f"flow_progress_{traj}" in group
                        and f"flow_signal_{traj}" in group
                        for traj in trajs
                    )
                print(
                    f"{path.name:90} groups={groups:6} "
                    f"trajs={total:8} flow={flow:8}/{total:<8} examples={examples}"
                )
        except Exception as exc:
            print(f"{path.name:90} ERROR {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
