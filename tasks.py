from celery import Celery


def get_search_for_batch(url):
    app = Celery("tasks", backend="rpc://", broker=f"{url}")

    # CELERY_ACKS_LATE = True
    app.conf.task_acks_late = True

    # 5 hrs timeout
    app.conf.update(broker_transport_options={"visibility_timeout": 18000},)

    @app.task
    def search_for_batch(batch_idx):
        pass

    return search_for_batch
