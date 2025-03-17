# main.py
import logging
import tensorflow as tf
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes
)
import numpy as np
from PIL import Image
import io
import os
import datetime

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

# Конфигурация
INPUT_SIZE = (150, 150)
NORMALIZATION = 1.0 / 255
THRESHOLD = 0.5
DATA_DIR = "training_data"
MODEL_PATH = "pipe_damaged_model.h5"
ADMIN_USER_ID = 1235678
YOUR_TOKEN = 123456

# Загрузка модели
model = tf.keras.models.load_model(MODEL_PATH)

# Логирование
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Создаем папку для данных
os.makedirs(DATA_DIR, exist_ok=True)


def process_image(image):
    """Обработка изображения для модели"""
    image = image.resize(INPUT_SIZE)
    image = np.array(image) * NORMALIZATION
    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)
    return np.expand_dims(image, axis=0)


async def ask_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE, prediction: str):
    """Запрос обратной связи"""
    keyboard = [
        [InlineKeyboardButton("Да", callback_data="correct"),
         InlineKeyboardButton("Нет", callback_data="wrong")]
    ]
    await update.message.reply_text(
        text=f"Результат: {prediction}\nПравильно ли я определил состояние трубы?",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


async def handle_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка ответа пользователя"""
    query = update.callback_query
    await query.answer()

    user_data = context.user_data
    if 'last_image' in user_data and 'last_prediction' in user_data:
        is_correct = query.data == "correct"
        await save_feedback(user_data['last_image'], is_correct, user_data['last_prediction'])
        await query.edit_message_text(text="Спасибо за обратную связь!")

        if len(os.listdir(DATA_DIR)) >= 10:
            await retrain_model()
    else:
        await query.edit_message_text(text="Ошибка: данные не найдены")


async def save_feedback(image: Image.Image, is_correct: bool, prediction: str):
    """Сохранение данных для обучения"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Создаем поддиректории для классов
    class_folder = "correct" if is_correct else "wrong"
    class_dir = os.path.join(DATA_DIR, class_folder)
    os.makedirs(class_dir, exist_ok=True)

    # Сохраняем изображение
    filename = os.path.join(class_dir, f"{timestamp}.png")
    image.save(filename)

    # Сохраняем метаданные с указанием класса
    with open(os.path.join(DATA_DIR, "metadata.csv"), "a", encoding="utf-8") as f:
        f.write(f"{filename},{class_folder},{prediction}\n")


async def retrain_model(force=False):
    """Дообучение модели"""
    try:
        # Проверка наличия данных
        class_dirs = ["correct", "wrong"]
        total_images = sum(
            [len(os.listdir(os.path.join(DATA_DIR, cls))) for cls in class_dirs]
        )
        if total_images == 0:
            raise ValueError("Нет данных для обучения.")

        # Загрузка и компиляция модели
        model = tf.keras.models.load_model(MODEL_PATH)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Подготовка данных
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            DATA_DIR,
            classes=class_dirs,
            target_size=INPUT_SIZE,
            batch_size=32,
            class_mode='binary'
        )

        # Дообучение
        model.fit(
            train_generator,
            epochs=3,
            verbose=1
        )

        # Сохранение модели
        model.save(MODEL_PATH)
        logger.info("Модель успешно дообучена")

    except Exception as e:
        logger.error(f"Ошибка дообучения: {str(e)}")
        raise

async def force_retrain(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Принудительный запуск переобучения"""
    try:
        # Проверка прав администратора
        if update.effective_user.id != ADMIN_USER_ID:
            await update.message.reply_text("❌ У вас нет прав на эту операцию")
            return

        await update.message.reply_text("🔄 Проверка данных и запуск переобучения...")
        await retrain_model(force=True)
        await update.message.reply_text("✅ Модель успешно переобучена!")

    except ValueError as ve:
        await update.message.reply_text(f"⚠️ Ошибка: {str(ve)}")
    except Exception as e:
        await update.message.reply_text("⚠️ Произошла непредвиденная ошибка. Проверьте логи.")
        logger.error(f"Ошибка принудительного переобучения: {e}")

async def handle_retrain_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    try:
        await query.edit_message_text("🔄 Начало переобучения...")
        await retrain_model(force=True)
        await query.edit_message_text("✅ Переобучение успешно завершено!")

    except Exception as e:
        await query.edit_message_text(f"❌ Ошибка: {str(e)}")

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка изображения"""
    try:
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = io.BytesIO()
        await photo_file.download_to_memory(out=photo_bytes)
        image = Image.open(photo_bytes)

        # Сохраняем данные в контексте
        context.user_data['last_image'] = image
        prediction = model.predict(process_image(image), verbose=0)[0][0]
        result = "не повреждена" if prediction > THRESHOLD else "повреждена"
        context.user_data['last_prediction'] = result

        await update.message.reply_text(f"Ваша труба {result}")
        await ask_feedback(update, context, result)

    except Exception as e:
        logger.error(f"Ошибка: {e}", exc_info=True)
        await update.message.reply_text("Ошибка обработки изображения")


async def start(update: Update, context) -> None:
    await update.message.reply_text('Привет! Отправь фото трубы для проверки.')


def main() -> None:
    # Проверка существования модели
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Модель {MODEL_PATH} не найдена!")
    # Проверка структуры директорий
    os.makedirs(os.path.join(DATA_DIR, "correct"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "wrong"), exist_ok=True)
    application = Application.builder().token(f"{YOUR_TOKEN}").build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("retrain", force_retrain))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    application.add_handler(CallbackQueryHandler(handle_feedback))
    application.add_handler(CallbackQueryHandler(handle_retrain_confirmation, pattern="^confirm_retrain$"))
    application.run_polling()


if __name__ == '__main__':
    main()
