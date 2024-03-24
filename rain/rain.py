import fire
import pandas as pd
import os
import requests
import PIL.Image
import io
import numpy as np
import easyocr
import datetime


def get_value_per_pixel_from_ocr_results(results):
    value_to_y_center = {}
    for result in results:
        bbox, text, score = result
        min_x, min_y = bbox[0]
        max_x, max_y = bbox[2]
        value = int(text)
        y_center = (min_y + max_y) / 2
        value_to_y_center[value] = y_center

    zero_point = value_to_y_center[0]
    max_val = max(value_to_y_center.keys())
    max_val_y_center = value_to_y_center[max_val]

    # Zero is down so has a larger y coordinate
    value_per_pixel = max_val / (zero_point - max_val_y_center)
    return value_per_pixel


def get_datetime(date_str):
    day, month, year = [int(i) for i in date_str.split("-")]
    return datetime.datetime(year, month, day)

def get_output_filename(output_dir, dt):
    date_string = dt.strftime('%Y-%m-%d')
    return os.path.join(output_dir, date_string + ".csv")


def fetch_rain_data(output_dir="./"):
    rain_img_url = "https://www.igf.fuw.edu.pl/m/meteo_station/WEBopad.png"

    result = requests.get(rain_img_url)
    img = PIL.Image.open(io.BytesIO(result.content))
    img = np.array(img)

    min_x = 51
    max_x = 626
    min_y = 31
    max_y = 202
    plot = img[min_y : max_y + 1, min_x : max_x + 1]

    # Calculate y scale
    y_axis_min_x = 26
    y_axis_max_x = 49
    y_axis_min_y = 21
    y_axis_max_y = 211
    y_axis_img = img[y_axis_min_y : y_axis_max_y + 1, y_axis_min_x : y_axis_max_x + 1]
    reader = easyocr.Reader(["en"], gpu=False)
    ocr_results = reader.readtext(
        y_axis_img, mag_ratio=3, min_size=3, allowlist="0123456789"
    )
    value_per_pixel = get_value_per_pixel_from_ocr_results(ocr_results)
    scale = value_per_pixel * (plot.shape[0] - 1)

    # Get date
    date_min_x = 280
    date_min_y = 2
    date_max_x = 393
    date_max_y = 26
    date_img = img[date_min_y : date_max_y + 1, date_min_x : date_max_x + 1]
    ocr_results = reader.readtext(
        date_img, mag_ratio=3, min_size=3, allowlist="0123456789-", detail=0
    )
    plot_datetime = get_datetime(ocr_results[0])

    red_mask = plot[..., 0] > plot[..., 1]

    y, x = np.where(red_mask)
    data_limit = np.max(x) + 1

    y_indices = np.argmax(red_mask, axis=0)

    max_y_index = red_mask.shape[0] - 1
    y_vals = (max_y_index - y_indices) * value_per_pixel
    x_vals = np.linspace(0, 24, red_mask.shape[1])
    x_vals = x_vals[:data_limit]
    y_vals = y_vals[:data_limit]

    utc_timestamp_offsets = x_vals * 3600
    utc_timestamps = plot_datetime.timestamp() + utc_timestamp_offsets

    filename = get_output_filename(output_dir=output_dir, dt=plot_datetime)
    df = pd.DataFrame({"precipitation_rate[mm/h]" : y_vals, "timestamp[UTC]" : utc_timestamps})
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    fire.Fire(fetch_rain_data)
