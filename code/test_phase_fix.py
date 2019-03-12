#!/usr/bin/env python

import numpy as np
import console
import cv2
import conversion

def main():
    stylized_img = np.load("stylized_img.npy")
    content_phase = np.load("content_phase.npy")
    for radius in [0.2, 0.5, 1, 1.5]:
        # stylized_image_sharp = unsharp_mask(stylized_img, radius=radius, amount=1)
        stylized_img_blur = cv2.GaussianBlur(stylized_img, (9,9), radius)
        stylized_img_sharp = cv2.addWeighted(stylized_img, 1.5, stylized_img_blur, -0.5, 0, stylized_img)
        stylized_audio = conversion.amplitude_to_audio(stylized_img_sharp, fft_window_size=1536, phase_iterations=15, phase=content_phase)
        conversion.audio_to_file(stylized_audio, "/Users/ollin/Desktop/stylized_random_phase.sharpened." + str(radius) + ".mp3")
        console.log("Tested radius", radius)

if __name__ == "__main__":
    main()
