from celery import Celery


def get_search_for_batch(url, backend_url):
    app = Celery("tasks", backend=f"{backend_url}", broker=f"{url}")

    # CELERY_ACKS_LATE = True
    app.conf.task_acks_late = True
    app.conf.worker_deduplicate_successful_tasks = True
    app.conf.worker_prefetch_multiplier = 1

    # 5 hrs timeout
    app.conf.update(broker_transport_options={"visibility_timeout": 18000})

    @app.task
    def search_for_batch(batch_idx):
        pass

    return search_for_batch
