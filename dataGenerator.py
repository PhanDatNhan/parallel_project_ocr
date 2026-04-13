import os
import random
import string
from PIL import Image, ImageDraw, ImageFont
import dataConfig


# ==============================
# Khởi tạo thư mục
# ==============================
os.makedirs(dataConfig.IMAGE_FOLDER, exist_ok=True)
os.makedirs(dataConfig.LABEL_FOLDER, exist_ok=True)


# ==============================
# Tạo chuỗi ký tự ngẫu nhiên
# ==============================
def generate_random_word(min_len=3, max_len=10):
    length = random.randint(min_len, max_len)
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


def generate_random_text_block(num_words=1000):
    words = [generate_random_word() for _ in range(num_words)]
    return " ".join(words)


# ==============================
# Chia text thành các dòng phù hợp
# ==============================
def wrap_text(text, draw, font, max_width):
    words = text.split(" ")
    lines = []
    current_line = ""

    for word in words:
        test_line = current_line + word + " "
        bbox = draw.textbbox((0, 0), test_line, font=font)
        line_width = bbox[2] - bbox[0]

        if line_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line.rstrip())
            current_line = word + " "

    if current_line:
        lines.append(current_line.rstrip())

    return lines


# ==============================
# Tạo nội dung đầy trang
# ==============================
def generate_full_page(draw, font, max_width, max_height):
    full_text = ""
    current_height = 0

    while current_height < max_height:
        raw_text = generate_random_text_block(200)
        lines = wrap_text(raw_text, draw, font, max_width)

        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_height = bbox[3] - bbox[1]

            if current_height + line_height > max_height:
                return full_text

            full_text += line + "\n"
            current_height += line_height + dataConfig.LINE_SPACING

    return full_text


# ==============================
# Render text thành ảnh
# ==============================
def render_image(text, image_path):
    img = Image.new(
        "RGB",
        (dataConfig.IMAGE_WIDTH, dataConfig.IMAGE_HEIGHT),
        "white"
    )

    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(dataConfig.FONT_PATH, dataConfig.FONT_SIZE)
    except:
        font = ImageFont.load_default()

    max_width = dataConfig.IMAGE_WIDTH - \
        dataConfig.MARGIN_LEFT - dataConfig.MARGIN_RIGHT
    max_height = dataConfig.IMAGE_HEIGHT - \
        dataConfig.MARGIN_TOP - dataConfig.MARGIN_BOTTOM

    wrapped_text = generate_full_page(draw, font, max_width, max_height)

    draw.multiline_text(
        (dataConfig.MARGIN_LEFT, dataConfig.MARGIN_TOP),
        wrapped_text,
        fill="black",
        font=font,
        spacing=dataConfig.LINE_SPACING
    )

    img.save(image_path)

    return wrapped_text


# ==============================
# Lưu ground truth
# ==============================
def save_label(text, label_path):
    with open(label_path, "w", encoding="utf-8") as f:
        f.write(text)


# ==============================
# Tạo dataset
# ==============================
def generate_dataset():
    print("Bắt đầu tạo dataset...")

    for i in range(dataConfig.NUM_IMAGES):
        image_name = f"img_{i:04d}.png"
        label_name = f"img_{i:04d}.txt"

        image_path = os.path.join(dataConfig.IMAGE_FOLDER, image_name)
        label_path = os.path.join(dataConfig.LABEL_FOLDER, label_name)

        text_content = render_image("", image_path)
        save_label(text_content, label_path)

        print(f"[{i+1}/{dataConfig.NUM_IMAGES}] Created {image_name}")

    print("Hoàn thành tạo dataset.")


# ==============================
# Main
# ==============================
if __name__ == "__main__":
    generate_dataset()
