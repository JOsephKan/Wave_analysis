# import pamonth_data
import numpy as np

# class for data processing
class pro:
    def __init__(self):
        self.sym_lat = None
        self.asy_lat = None
        self.lat_sum = None
        self.weight_sym = None
        self.weight_asy = None
        self.avg_sym = None
        self.avg_asy = None
        self.time_range = None
        self.split_data = None

    def sym_format(self, lat, data):
        sym_lat = np.cos(np.deg2rad(lat))
        self.lat_sum = np.sum(sym_lat)
        
        weighted = data * sym_lat[None, :, None]

        avg_sym = np.sum(weighted, axis=1) / self.lat_sum

        return avg_sym

    def asy_format(self, lat, data):
        if lat[0] < 0.0:
            lat = lat[::-1]
            data = data[:, ::-1, :]

        asy_lat = np.empty_like(lat)
        asy_lat[: int(lat.shape[0] / 2)] = np.cos(
            np.deg2rad(lat[: int(lat.shape[0] / 2)])
        )
        asy_lat[int(lat.shape[0] / 2) :] = -np.cos(
            np.deg2rad(lat[int(lat.shape[0] / 2) :])
        )

        weighted = data * asy_lat[None, :, None]

        self.avg_asy = np.sum(weighted, axis=1) / self.lat_sum

        return self.avg_asy

def background(data, num_of_passes):
    padded_data = np.empty((data.shape[0], data.shape[1] + 2), dtype=data.dtype)
    padded_data[:, 0] = data[:, 1]
    padded_data[:, -1] = data[:, -2]
    padded_data[:, 1:-1] = data

    for k in range(num_of_passes):
        for j in range(padded_data.shape[0]):
            for i in range(1, padded_data.shape[1] - 1):
                padded_data[j, i] = (
                    padded_data[j, i - 1]
                    + padded_data[j, i] * 2
                    + padded_data[j, i + 1]
                ) / 4

    return padded_data[:, 1:-1]

