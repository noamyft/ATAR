from GPSPhoto import gpsphoto
import geopy.distance
import os
from datetime import datetime
import pandas as pd
import cv2

def crop_resize_folder(base_folder, selected_folder, destination_path):
    dir = base_folder + "/" + selected_folder
    photos = os.listdir(dir)
    os.makedirs(destination_path, exist_ok=True)

    for p in photos:
        img = cv2.imread(dir + "/" + p)
        crop_img = img[40:1120]
        resize_img = cv2.resize(crop_img, (224, 224), interpolation=cv2.INTER_AREA)
        cv2.imwrite(destination_path + "/" + selected_folder + "_" + p, resize_img)

def process_folder(base_folder, selected_folder):

    dir = base_folder + "/" + selected_folder
    photos = os.listdir(dir)
    photos.sort()
    result = []

    for i in range(1, len(photos)-1):

        data1 = gpsphoto.getGPSData(dir + '/' + photos[i-1])
        data2 = gpsphoto.getGPSData(dir + '/' + photos[i])

        coords_1 = (data1['Latitude'], data1['Longitude'])
        coords_2 = (data2['Latitude'], data2['Longitude'])
        distance_diff = geopy.distance.vincenty(coords_1, coords_2).km

        time_1 = datetime.strptime(data1['UTC-Time'], '%H:%M:%S')
        time_2 = datetime.strptime(data2['UTC-Time'], '%H:%M:%S')
        time_diff = (time_2 - time_1).seconds

        speed = None if time_diff > 10 else distance_diff * 60 * 60 / time_diff

        result.append((photos[i], selected_folder, data2['Latitude'], data2['Longitude'],
                       data2['UTC-Time'], data2['Date'], speed))

    return result, ["photo", "dir", "latitude", "longitude", "time", "date", "speed"]

def main():
    base_folder = 'raw_data'
    selected_folder = 'd01'

    destination_path = "data/small"
    data, cols = process_folder(base_folder, selected_folder)

    df = pd.DataFrame(data,columns=cols)

    # remove nan
    df = df[df['speed'].notnull()]
    # save data
    df.to_csv("data.csv")


    # process images
    # crop_resize_folder(base_folder, selected_folder, destination_path)





if __name__ == '__main__':
    main()