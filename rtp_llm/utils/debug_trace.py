import functools
from contextlib import contextmanager

import viztracer


@contextmanager
def trace_scope(name):
    tracer = viztracer.VizTracer(
        tracer_entries=2000000, log_gc=True, register_global=False
    )
    tracer.start()
    start_time = tracer.getts()
    try:
        yield
    finally:
        end_time = tracer.getts()
        tracer.stop()
        if name:
            tracer.save(
                f"/home/caihaowen.chw/work/RTP-LLM/github-opensource/{name}.json"
            )


def trace_func(name_gen=lambda f, *args, **kwargs: f.__name__):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            name = name_gen(f, *args, **kwargs) if name_gen is not None else None
            with trace_scope(name):
                return f(*args, **kwargs)

        return wrapper

    return decorator
