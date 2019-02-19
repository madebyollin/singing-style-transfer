#!/usr/bin/env python
import conversion
import console
import numpy as np
from post_processor import PostProcessor

post_processor = PostProcessor()
post_processor.load_weights("weights.h5")

stylized = conversion.file_to_image("sample/rolling_in_the_deep/stylized.png")
content_harmonics = conversion.file_to_image("sample/rolling_in_the_deep/content.mp3.harmonics.png")
content_sibilants = conversion.file_to_image("sample/rolling_in_the_deep/content.mp3.harmonics.png")

stylized = post_processor.predict_unstacked(amplitude=stylized, harmonics=content_harmonics, sibilants=content_sibilants)

conversion.image_to_file(stylized, "/Users/ollin/Desktop/boop.png")
