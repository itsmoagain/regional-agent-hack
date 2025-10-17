import argparse

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--region', required=True, choices=['hungary_farmland','jamaica_coffee'])
    args = p.parse_args()
    # TODO: orchestrate fetch scripts per region
    print(f"[demo] building cache for {args.region}")
