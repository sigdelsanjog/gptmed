# framework package init
from .logging_utils import get_framework_logs_dir

# Ensure logs directory exists when framework is imported
get_framework_logs_dir()
