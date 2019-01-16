#!/usr/bin/env python
import conversion
import console

style_path = "sample/style.mp3"
content_path = "sample/content.mp3"
stylized_path = "sample/stylized.mp3"

style_img = conversion.file_to_image(style_path)
content_img = conversion.file_to_image(content_path)

def stylize(style, content):
    return content

stylized = stylize(style_img, content_img)

conversion.image_to_file(stylized, stylized_path)
console.log("done! saved to", stylized_path)
