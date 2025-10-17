from codecarbon import EmissionsTracker

def measure(fn, *args, **kwargs):
    tracker = EmissionsTracker()
    tracker.start()
    result = fn(*args, **kwargs)
    emissions = tracker.stop()
    return result, emissions
