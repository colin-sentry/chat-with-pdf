import time

import sentry_sdk
from sentry_sdk.ai.monitoring import ai_track, record_token_usage


sentry_sdk.init(
    # dsn="http://d3d3ffcc17c9729ae3fffaa90d97ce02@localhost:3001/9",
    dsn="https://8f3b03f3d70993ccc67a99fdd2276271@o1.ingest.us.sentry.io/4506893166379008", # https://sentry.sentry.io/settings/projects/ask-sentry/keys/
    enable_tracing=True,
    traces_sample_rate=1.0,
    send_default_pii=True,
    debug=True
)

@ai_track(description="My op")
def ai_op(id: int):
    with sentry_sdk.start_span(description="LLM call", op="ai.chat_completions.create.openai") as span:
        record_token_usage(span, prompt_tokens=10, completion_tokens=20, total_tokens=30)
        span.set_data("ai.model_id", "gpt-4")
        span.set_data("ai.streaming", False)
        span.set_data("test.id", id)

@ai_track(description="Colin pipeline")
def pipeline(id: int):
    ai_op(id)


for i in range(10):
    print("Sending pipeline.")
    with sentry_sdk.start_transaction():
        pipeline(i)
    time.sleep(10)
