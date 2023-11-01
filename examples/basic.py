from microcore import llm

while user_msg := input('Enter message: '):
    print('AI: ' + llm(user_msg))
