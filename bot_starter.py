import networkx
import telebot
from telebot import types
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel
import os

from searchformeaning.summarizer.sfm import SearchForMeaning
import bot_config

tokenizer_rubert = AutoTokenizer.from_pretrained(
    os.path.abspath("searchformeaning/summarizer/models/bert/rubert_cased_L-12_H-768_A-12_pt_v1"))
model_rubert = AutoModel.from_pretrained(os.path.abspath(
    "searchformeaning/summarizer/models/bert/rubert_cased_L-12_H-768_A-12_pt_v1"))
model_w2v = Word2Vec.load(os.path.abspath('searchformeaning/summarizer/models/word2vec/w2v.model'))

#  Connection to the bot
bot = telebot.TeleBot(bot_config.TOKEN)

#  Default settings
IMPORTANCE_ALGORITHM = True
TEXTRANK_ALGORITHM = True
LEXRANK_ALGORITHM = True
LSA_ALGORITHM = True
LUHN_ALGORITHM = True
MINTO_ALGORITHM = True
FEATURES_ALGORITHM = True
WORD2VEC_ALGORITHM = True
RUBERT_ALGORITHM = True
EXTRACT_PERCENTAGE = 75


@bot.message_handler(commands=['start', 'help'])
def start_bot(message: types.Message) -> None:
    """
    'Welcome' function.

    :param message: (Message)
        Telegram message.
    :return: (None)
    """
    bot.delete_message(message.chat.id, message.message_id)
    markup = types.InlineKeyboardMarkup(row_width=2)
    show_settings = types.InlineKeyboardButton("Показать текущие настройки", callback_data='show_configure')
    settings = types.InlineKeyboardButton("Настроить бота", callback_data='configure')
    show_bot_info = types.InlineKeyboardButton("Показать информацию о боте", callback_data='show_bot_info')
    markup.add(show_settings, show_bot_info, settings)

    bot.send_message(message.chat.id, "Добро пожаловать! Это бот, выполняющий реферирование публицистических текстов на"
                                      " русском языке. \n\nНастройте бота (если не настроить - будут использованы "
                                      "настройки по умолчанию) или отправьте текст для суммаризации.",
                     reply_markup=markup)


@bot.message_handler(content_types=['text'])
def summarize(message: types.Message) -> None:
    """
    The function that summarizes the incoming text.

    :param message: (Message)
        Telegram message that consist a text for summarization.
    :return: None.
    """

    try:
        if message.chat.type == 'private':
            model = SearchForMeaning(message.text, EXTRACT_PERCENTAGE)
            scores = [0] * len(model.raw_sentences)

            if IMPORTANCE_ALGORITHM:
                for snt in model.use_importance_algorithm():
                    scores[snt[0]] += 1

            if TEXTRANK_ALGORITHM:
                for snt in model.use_textrank():
                    scores[snt[0]] += 1

            if LEXRANK_ALGORITHM:
                for snt in model.use_lexrank():
                    scores[snt[0]] += 1

            if LSA_ALGORITHM:
                for snt in model.use_lsa():
                    scores[snt[0]] += 1

            if LUHN_ALGORITHM:
                for snt in model.use_luhn():
                    scores[snt[0]] += 1

            if MINTO_ALGORITHM:
                for snt in model.use_minto():
                    scores[snt[0]] += 1

            if FEATURES_ALGORITHM:
                for snt in model.use_features_algorithm():
                    scores[snt[0]] += 1

            if WORD2VEC_ALGORITHM:
                for snt in model.use_word2vec(model_w2v=model_w2v):
                    scores[snt[0]] += 1

            if RUBERT_ALGORITHM:
                for snt in model.use_rubert(model_rubert=model_rubert, tokenizer_rubert=tokenizer_rubert):
                    scores[snt[0]] += 1

            bot.send_message(message.chat.id, ' '.join([row[1] for row in model.extract_sentences(scores)])
                             .replace('  ', ' '))
    except ValueError:
        bot.send_message(message.chat.id, 'Ошибка! Текст должен удовлетворять следующим условиям:\n'
                                          '1. Больше чем 50 кириллических символов.\n'
                                          '2. Больше чем 3 предложения.\n'
                                          '3. Нет предложений, содержащих только латинские буквы.\n'
                                          '4. Нет EMOJI.')

    except networkx.exception.PowerIterationFailedConvergence:
        bot.send_message(message.chat.id, 'Ошибка! Недопустимый текст.')


@bot.callback_query_handler(func=lambda call: True)
def callback_inline(call: types.CallbackQuery) -> None:
    """
    The function that performs callback queries.

    :param call: (CallbackQuery)
        Callback query.
    :return: None.
    """
    if call.message:
        global WORD2VEC_ALGORITHM, RUBERT_ALGORITHM, EXTRACT_PERCENTAGE
        if call.data == 'show_configure':
            markup_back = types.InlineKeyboardMarkup(row_width=1)
            back = types.InlineKeyboardButton("Назад", callback_data='back_button')
            markup_back.add(back)
            bot.send_message(
                call.message.chat.id,
                f'Процент экстракции: {EXTRACT_PERCENTAGE}%.\nРежим: '
                f'{"Медленный." if WORD2VEC_ALGORITHM == True and RUBERT_ALGORITHM == True else "Быстрый."}',
                reply_markup=markup_back
            )

        elif call.data == 'configure':
            markup_percentage = types.InlineKeyboardMarkup(row_width=2)
            fifty_percent = types.InlineKeyboardButton("50%", callback_data='fifty_percent')
            seventy_five_percent = types.InlineKeyboardButton("75%", callback_data='seventy_five_percent')
            another_value = types.InlineKeyboardButton("Ввести другое число", callback_data='another_value')
            markup_percentage.add(fifty_percent, seventy_five_percent, another_value)
            bot.send_message(call.message.chat.id, 'Выберите процент экстракции:',
                             reply_markup=markup_percentage)

        elif call.data == 'show_bot_info':
            markup_back = types.InlineKeyboardMarkup(row_width=1)
            back = types.InlineKeyboardButton("Назад", callback_data='back_button')
            markup_back.add(back)
            bot.send_message(call.message.chat.id,
                             f'Интеллектуальная система поиска основного смысла в тексте публицистического содержания,'
                             f' 2023.'
                             f'\n\nАвтор: Зарипов Шамиль \nEmail: mail@tonyfierro.com \nTelegram: @tonyfierro',
                             reply_markup=markup_back)

        if call.data == 'fast_mode':
            bot.delete_message(call.message.chat.id, call.message.message_id)
            bot.answer_callback_query(callback_query_id=call.id, show_alert=False,
                                      text="Выбран быстрый режим.")
            WORD2VEC_ALGORITHM = False
            RUBERT_ALGORITHM = False

            bot.answer_callback_query(callback_query_id=call.id, show_alert=False,
                                      text="Настройка завершена!")
            bot.send_message(call.message.chat.id, 'Настройки изменены!')

        elif call.data == 'slow_mode':
            bot.delete_message(call.message.chat.id, call.message.message_id)
            bot.answer_callback_query(callback_query_id=call.id, show_alert=False,
                                      text="Выбран медленный режим.")
            WORD2VEC_ALGORITHM = True
            RUBERT_ALGORITHM = True

            bot.answer_callback_query(callback_query_id=call.id, show_alert=False,
                                      text="Настройка завершена!")
            bot.send_message(call.message.chat.id, 'Настройки изменены!')

        if call.data == 'fifty_percent':
            bot.delete_message(call.message.chat.id, call.message.message_id)
            bot.answer_callback_query(callback_query_id=call.id, show_alert=False,
                                      text="Коэффициент экстракции: 50%.")
            EXTRACT_PERCENTAGE = 50

            markup_mode = types.InlineKeyboardMarkup(row_width=2)
            fast_mode = types.InlineKeyboardButton("Быстрый, но менее точный", callback_data='fast_mode')
            slow_mode = types.InlineKeyboardButton("Медленный, но более точный", callback_data='slow_mode')
            markup_mode.add(fast_mode, slow_mode)
            bot.send_message(call.message.chat.id, 'Выберите режим:', reply_markup=markup_mode)

        elif call.data == 'seventy_five_percent':
            bot.delete_message(call.message.chat.id, call.message.message_id)
            bot.answer_callback_query(callback_query_id=call.id, show_alert=False,
                                      text="Коэффициент экстракции: 75%.")
            EXTRACT_PERCENTAGE = 75

            markup_mode = types.InlineKeyboardMarkup(row_width=2)
            fast_mode = types.InlineKeyboardButton("Быстрый, но менее точный", callback_data='fast_mode')
            slow_mode = types.InlineKeyboardButton("Медленный, но более точный", callback_data='slow_mode')
            markup_mode.add(fast_mode, slow_mode)
            bot.send_message(call.message.chat.id, 'Выберите режим:', reply_markup=markup_mode)

        elif call.data == 'another_value':
            msg = bot.send_message(call.message.chat.id, 'Введите процент экстракции (диапазон от 1 до 100):')
            bot.register_next_step_handler(msg, change_extract_percentage)

        if call.data == 'back_button':
            bot.delete_message(call.message.chat.id, call.message.message_id)


def choose_mode(message: types.Message) -> None:
    """
    This function is used to send a message that offers to select an execution mode.

    :param message: (types.Message)
        Telegram message.
    :return: (None)
    """

    markup_mode = types.InlineKeyboardMarkup(row_width=2)
    fast_mode = types.InlineKeyboardButton("Быстрый, но менее точный", callback_data='fast_mode')
    slow_mode = types.InlineKeyboardButton("Медленный, но более точный", callback_data='slow_mode')
    markup_mode.add(fast_mode, slow_mode)
    bot.send_message(message.chat.id, 'Выберите режим:', reply_markup=markup_mode)


def change_extract_percentage(message: types.Message) -> None:
    """
    This function is used to change the extract coefficient.

    :param message: (types.Message)
        Telegram message.
    :return: (None)
    """

    global EXTRACT_PERCENTAGE
    try:
        if 1 <= int(message.text) <= 100:
            EXTRACT_PERCENTAGE = int(message.text)
            choose_mode(message)
        else:
            msg = bot.send_message(message.chat.id, 'Вводимое число должно соответствовать диапазону от 1 до 100. '
                                                    'Значение коэффициента экстракции не изменилось.'
                                                    'Введите новое значение:')
            bot.register_next_step_handler(msg, change_extract_percentage)
    except ValueError:
        msg = bot.send_message(message.chat.id, 'Ошибка ввода! Значение коэффициента экстракции не изменилось. '
                                                'Введите новое значение:')
        bot.register_next_step_handler(msg, change_extract_percentage)


bot.polling(none_stop=True)
