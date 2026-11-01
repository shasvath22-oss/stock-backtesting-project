import argparse
import sys
from pathlib import Path

from .common import begin_stage, create_run
from .data import collect
from .features import features
from .modeling import predict, splits, train
from .backtest import backtest
from .reporting import report


def execute(stage, run, horizon=None, bps=None):
    begin_stage(run,stage,horizon)
    if stage=="train":
        train(run,horizon)
    elif stage=="backtest":
        backtest(run,bps)
    else:
        {"collect":collect,"features":features,"splits":splits,"predict":predict,"report":report}[stage](run)


def main():
    parser = argparse.ArgumentParser(description="Market direction research pipeline")
    parser.add_argument("stage",choices=["init","collect","features","splits","train","predict","backtest","report","run-all"])
    parser.add_argument("--config",default="config/research.json")
    parser.add_argument("--run-id",help="Artifact directory name; required after init/collect")
    parser.add_argument("--full",action="store_true",help="Use all 50 configured instruments")
    parser.add_argument("--learning",action="store_true",help="Create an AAPL-only run with a 5-session horizon")
    parser.add_argument("--horizon",type=int)
    parser.add_argument("--slippage-bps",type=int)
    args = parser.parse_args()
    try:
        if args.run_id:
            # Disallow arbitrary paths in CLI/UI run selection.
            if Path(args.run_id).name != args.run_id:
                raise ValueError("run-id must be an artifact folder name")
            run = Path("artifacts")/args.run_id
            from .common import manifest
            if manifest(run).get("synthetic",True):
                raise ValueError("Only yfinance historical experiments are supported; create a new historical run")
        elif args.stage in ["init","collect","run-all"]:
            run = create_run(args.config,demo=False,full=args.full,learning=args.learning)
        else:
            raise ValueError("Supply --run-id from an initialized run")
        print(f"Run: {run}",flush=True)
        if args.stage=="init":
            return
        stages = ["collect","features","splits","train","predict","backtest","report"] if args.stage=="run-all" else [args.stage]
        for stage in stages:
            print(f"Starting {stage}...",flush=True)
            execute(stage,run,args.horizon,args.slippage_bps)
            print(f"Completed {stage}",flush=True)
    except Exception as exc:
        print(f"ERROR: {exc}",file=sys.stderr)
        sys.exit(1)


if __name__=="__main__":
    main()
