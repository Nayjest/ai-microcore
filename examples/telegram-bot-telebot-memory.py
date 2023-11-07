"""
Example of a Telegram Bot with shot memory (shared among all dialogues).
Requirements:
- pip install pyTelegramBotAPI
- provide a TELEGRAM_BOT_TOKEN environment variable (you may use .env file)
"""
import os
from collections import deque
import telebot
import microcore as mc

mc.configure()
mc.use_logging()
bot = telebot.TeleBot(os.getenv('TELEGRAM_BOT_TOKEN'))

memory = deque(maxlen=5)
sys_msg = mc.SysMsg("You are the pirate. Yarrr! Be rude, use obscene internet language")


@bot.message_handler(func=lambda message: True)
def echo_all(message):
    memory.append(mc.UserMsg(message.text))
    bot_msg = mc.AssistantMsg(mc.llm([sys_msg, *memory]))
    memory.append(bot_msg)
    bot.reply_to(message, bot_msg.content)


bot.infinity_polling()
