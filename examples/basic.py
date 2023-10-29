import microcore
while user_msg := input('Enter message: '):
    print('AI: ' + microcore.llm(user_msg))
