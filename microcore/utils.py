import os


def is_chat_model(model: str) -> bool:
    completion_keywords = ['instruct', 'davinci', 'babbage', 'curie', 'ada']
    return not any(keyword in model for keyword in completion_keywords)


true_values = ['1', 'TRUE', 'YES', 'ON', 'ENABLED']


def get_bool_from_env(env_var: str, default: bool = False):
    return os.getenv(env_var, str(default)).upper() in true_values
