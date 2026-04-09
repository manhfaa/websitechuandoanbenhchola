class AppError(Exception):
    status_code = 500


class BadRequestError(AppError):
    status_code = 400


class ConfigurationError(AppError):
    status_code = 500


class DependencyError(AppError):
    status_code = 500


class InferenceError(AppError):
    status_code = 500
