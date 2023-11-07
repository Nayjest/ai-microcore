"""
This is a simple example of a Telegram Bot that receives messages and replies to them.
Requirements:
- pip install pyTelegramBotAPI
- provide a TELEGRAM_BOT_TOKEN environment variable (you may use .env file)
"""
import os
import telebot
import microcore as mc


mc.configure()
bot = telebot.TeleBot(os.getenv('TELEGRAM_BOT_TOKEN'))


@bot.message_handler(func=lambda message: True)
def echo_all(message):
    bot.reply_to(message, mc.llm(message.text))


bot.infinity_polling()
