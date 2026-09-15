import warnings
import logging
import structlog


warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*analyzers are deprecated.*"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*model profiles are deprecated.*"
)

logging.getLogger("evidently").setLevel(logging.WARNING)

structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(colors=True)
        ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.NOTSET),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False
)