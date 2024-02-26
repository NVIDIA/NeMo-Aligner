from celery import Celery

app = Celery("tasks", backend="rpc://", broker="pyamqp://guest@10.110.40.145:5672//")

# CELERY_ACKS_LATE = True
app.conf.task_acks_late = True


@app.task
def search_for_batch(batch_idx):
    pass
