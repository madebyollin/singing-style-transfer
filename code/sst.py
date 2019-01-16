#!/usr/bin/env python
import conversion
import console

def global_eq_match(content, style):
    return content

def stylize(content, style):
    stylized = global_eq_match(content, style)
    return stylized

def main():
    style_path = "sample/style.mp3"
    content_path = "sample/content.mp3"
    stylized_path = "sample/stylized.mp3"

    style_img = conversion.file_to_image(style_path)
    content_img = conversion.file_to_image(content_path)
    stylized = stylize(content_img, style_img)

    conversion.image_to_file(stylized, stylized_path)
    console.log("done! saved to", stylized_path)

if __name__ == "__main__":
    main()
