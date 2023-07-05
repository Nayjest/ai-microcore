from microcore import *

def logging(): import microcore.logging
def model(name: str): microcore.llm_default_args['model'] = name