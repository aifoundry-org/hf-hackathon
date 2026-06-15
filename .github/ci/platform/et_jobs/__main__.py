import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m et_jobs api|worker|runner", file=sys.stderr)
        sys.exit(1)
    cmd = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    if cmd == "api":
        from .api import main

        main()
    elif cmd == "worker":
        from .worker import main

        main()
    elif cmd == "runner":
        from .runner import main

        sys.exit(main())
    else:
        print(f"unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)
