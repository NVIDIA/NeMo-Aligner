from celery import Celery


def get_search_for_batch(url):
    app = Celery("tasks", backend="rpc://", broker=f"pyamqp://guest@{url}//")

    # CELERY_ACKS_LATE = True
    app.conf.task_acks_late = True

    @app.task
    def search_for_batch(batch_idx):
        pass

    return search_for_batch
