"""
pip install aiogram
"""
import asyncio
import logging
import os
from collections import deque
from pprint import pprint

from aiogram import Bot, Dispatcher, types
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramRetryAfter
import microcore as mc
from microcore import Msg

mc.use_logging()
dp = Dispatcher()

_memory: dict[str, deque] = {}


def get_short_memory(chat_id) -> deque[Msg]:
    return _memory.setdefault(chat_id, deque(maxlen=5))


SYS_MSG = mc.SysMsg(
    os.getenv(
        "BOT_SYS_MSG",
        "You are the pirate. Yarrr! Be rude, use obscene internet language.",
    )
)
NOTHING_TO_SAY = os.getenv("BOT_NOTHING_TO_SAY", "Nothing to say.")
YOU = os.getenv("BOT_YOU", "You")
MEMORIES_MSG = os.getenv("BOT_MEMORIES_MSG", "Some memories from the past:") + "\n"


@dp.message()
async def echo_handler(tg_msg: types.Message):
    chunks = []
    finished = False

    async def ask_llm():
        nonlocal finished, chunks
        user_msg = mc.UserMsg(f"@{tg_msg.from_user.username}: {tg_msg.text}")
        short_memory = get_short_memory(tg_msg.chat.id)
        long_memory_id = f"chat_{tg_msg.chat.id}"
        memories = mc.texts.search(long_memory_id, user_msg.content, n_results=3)
        pprint(memories)
        pprint(tg_msg.chat.id)
        composed_prompt = [SYS_MSG]
        if memories:
            composed_prompt += [mc.SysMsg(MEMORIES_MSG + "\n".join(memories))]
        composed_prompt += [*short_memory, user_msg]
        ai_response = await mc.allm(composed_prompt, callback=chunks.append)
        finished = True
        short_memory += [user_msg, mc.AssistantMsg(ai_response)]
        long_memory_record = f"{user_msg.content}\n{YOU}:{ai_response}"
        mc.texts.save(long_memory_id, long_memory_record)

    async def tele_send_to_user(chat_id):
        nonlocal finished, chunks
        logging.info("bot.send_message... ")
        ai_msg = await bot.send_message(chat_id=chat_id, text="...")
        await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        logging.info("entering message update loop... ")
        last_iteration = False
        while True:
            text = "".join(chunks)
            if text and text != ai_msg.text:
                try:
                    await ai_msg.edit_text(text)
                    if not finished:
                        await bot.send_chat_action(
                            chat_id=chat_id, action=ChatAction.TYPING
                        )
                except TelegramRetryAfter as e:
                    logging.error(e)
                    await asyncio.sleep(10)
                    continue
                except Exception as e:
                    logging.error(e)
            if last_iteration:
                if not text:
                    await ai_msg.edit_text(NOTHING_TO_SAY)

                logging.info("finished updating message")
                break
            if finished:
                last_iteration = True
            else:
                await asyncio.sleep(0.2)

    await asyncio.gather(ask_llm(), tele_send_to_user(tg_msg.chat.id))


logging.basicConfig(level=logging.INFO)
bot = Bot(os.getenv("TELEGRAM_BOT_TOKEN"))
asyncio.run(dp.start_polling(bot))
