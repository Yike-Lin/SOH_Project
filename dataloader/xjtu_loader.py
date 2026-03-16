import os
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from utils.scaler import Scaler


class XJTUDataset:
    """XJTU Battery Dataset Loader"""
    
    def __init__(self, args):
        self.root = 'data/XJTU'
        self.max_capacity = 2.0
        self.normalized_type = args.normalized_type
        self.minmax_range = args.minmax_range
        self.seed = args.random_seed
        self.batch = getattr(args, 'batch', None)
        self.batch_size = args.batch_size

    def _row(self, x):
        """Reshape array to (1, -1) for sequence concatenation"""
        return np.asarray(x).squeeze().reshape(1, -1)

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize input data based on initialized scaler settings"""
        scaler = Scaler(data)
        if self.normalized_type == 'standard':
            return scaler.standard
        return scaler.minmax(feature_range=self.minmax_range)
    
    def _parser_mat_data(self, battery_i_mat):
        """Parse partial/time-series data from a single battery .mat struct"""
        data_list, cap_list = [], []
        
        for i in range(battery_i_mat.shape[1]):
            cycle_i = battery_i_mat[0, i]
            time = cycle_i['relative_time_min']
            current = cycle_i['current_A']
            voltage = cycle_i['voltage_V']
            temp = cycle_i['temperature_C']
            capacity = cycle_i['capacity'][0]

            cap_list.append(capacity)
            cycle_arr = np.concatenate([time, current, voltage, temp], axis=0)
            data_list.append(cycle_arr)

        data = np.asarray(data_list, dtype=np.float32)
        label = np.asarray(cap_list, dtype=np.float32)

        data = self._normalize(data)
        soh = label / self.max_capacity

        return data, soh
    
    def _parser_full_cycle(self, battery_i_mat):
        """Parse full charge and discharge data from a single battery .mat struct"""
        data_list, cap_list = [], []
        num_cycles = battery_i_mat.shape[1]

        for i in range(num_cycles):
            cyc = battery_i_mat[0, i]

            # Extract Capacity
            if 'capacity' in cyc.dtype.names:
                capacity = cyc['capacity'][0]
            elif 'cycle' in cyc.dtype.names:
                cyc2 = cyc['cycle'][0, 0]
                if isinstance(cyc2, np.void) and 'capacity' in cyc2.dtype.names:
                    capacity = cyc2['capacity'].item()
                else:
                    raise KeyError("Missing capacity field in nested cycle struct.")
            else:
                raise KeyError("Missing capacity field in .mat structure.")
            
            cap_list.append(float(np.asarray(capacity).squeeze()))

            # Extract Charge Data
            ch = cyc['charge_data'][0, 0]
            ch_data = [
                self._row(ch['relative_time_min']), self._row(ch['current_A']),
                self._row(ch['voltage_V']), self._row(ch['temperature_C'])
            ]

            # Extract Discharge Data
            dis = cyc['discharge_data'][0, 0]
            dis_data = [
                self._row(dis['relative_time_min']), self._row(dis['current_A']),
                self._row(dis['voltage_V']), self._row(dis['temperature_C'])
            ]

            cycle_arr = np.concatenate(ch_data + dis_data, axis=0)
            data_list.append(cycle_arr)

        data = np.asarray(data_list, dtype=np.float32)
        label = np.asarray(cap_list, dtype=np.float32)

        data = self._normalize(data)
        soh = label / self.max_capacity

        return data, soh
    
    def _encapsulation(self, train_x, train_y, test_x, test_y):
        """Convert numpy arrays to PyTorch DataLoaders with a train/val split"""
        train_x = torch.from_numpy(train_x)
        train_y = torch.from_numpy(train_y).view(-1, 1)
        test_x = torch.from_numpy(test_x)
        test_y = torch.from_numpy(test_y).view(-1, 1)

        tr_x, val_x, tr_y, val_y = train_test_split(
            train_x, train_y, test_size=0.2, random_state=self.seed
        )

        train_loader = DataLoader(TensorDataset(tr_x, tr_y), batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=self.batch_size, shuffle=False)

        return train_loader, valid_loader, test_loader
    
    def _get_raw_data(self, path, test_battery_id):
        """Read .mat file and perform leave-one-battery-out splitting"""
        mat = loadmat(path)
        battery = mat['battery']
        num_batt = battery.shape[1]
        battery_ids = list(range(1, num_batt + 1))

        if test_battery_id not in battery_ids:
            raise IndexError(f'test_battery_id must be in {battery_ids}, got {test_battery_id}')

        test_battery = battery[0, test_battery_id - 1][0]
        test_x, test_y = self._parser_mat_data(test_battery)

        train_x_list, train_y_list = [], []
        for bid in battery_ids:
            if bid == test_battery_id:
                continue
            x, y = self._parser_mat_data(battery[0, bid - 1][0])
            train_x_list.append(x)
            train_y_list.append(y)

        train_x = np.concatenate(train_x_list, axis=0)
        train_y = np.concatenate(train_y_list, axis=0)

        return self._encapsulation(train_x, train_y, test_x, test_y)
    
    def _get_full_raw_data(self, path, test_battery_id):
        """Read full-cycle .mat file and perform leave-one-battery-out splitting"""
        mat = loadmat(path)
        battery = mat['battery']
        num_batt = battery.shape[1]
        battery_ids = list(range(1, num_batt + 1))

        if test_battery_id not in battery_ids:
            raise IndexError(f'test_battery_id must be in {battery_ids}, got {test_battery_id}')
        
        test_battery = battery[0, test_battery_id - 1][0]
        test_x, test_y = self._parser_full_cycle(test_battery)

        train_x_list, train_y_list = [], []
        for bid in battery_ids:
            if bid == test_battery_id:
                continue
            x_i, y_i = self._parser_full_cycle(battery[0, bid - 1][0])
            train_x_list.append(x_i)
            train_y_list.append(y_i)
        
        train_x = np.concatenate(train_x_list, axis=0)
        train_y = np.concatenate(train_y_list, axis=0)

        return self._encapsulation(train_x, train_y, test_x, test_y)

    def get_charge_data(self, test_battery_id=1):
        """Public API: Load full charge curves"""
        path = os.path.join(self.root, 'charge', f'batch-{self.batch}.mat')
        loaders = self._get_raw_data(path, test_battery_id)
        return dict(zip(['train', 'valid', 'test'], loaders))
    
    def get_partial_data(self, test_battery_id=1):
        """Public API: Load partial charge curves"""
        suffix = '3.9-4.19' if self.batch == 6 else '3.7-4.1'
        path = os.path.join(self.root, 'partial_charge', f'batch-{self.batch}_{suffix}.mat')
        loaders = self._get_raw_data(path, test_battery_id)
        return dict(zip(['train', 'valid', 'test'], loaders))

    def _parser_xlsx(self, df_i: pd.DataFrame):
        """Parse features and labels from excel sheets"""
        x = np.asarray(df_i.iloc[:, :-1], dtype=np.float32)
        label = np.asarray(df_i['label'], dtype=np.float32)
        x = self._normalize(x)
        return x, label / self.max_capacity
    
    def get_features(self, test_battery_id=1):
        """Public API: Load handcrafted features from excel"""
        path = os.path.join(self.root, 'handcraft_features', f'batch-{self.batch}_features.xlsx')
        df_dict = pd.read_excel(path, sheet_name=None)
        sheet_names = list(df_dict.keys())
        battery_ids = list(range(1, len(sheet_names) + 1))

        if test_battery_id not in battery_ids:
            raise IndexError(f'test_battery_id must be in {battery_ids}, got {test_battery_id}')

        test_df = pd.read_excel(path, sheet_name=test_battery_id - 1, header=0)
        test_x, test_y = self._parser_xlsx(test_df)

        train_x_list, train_y_list = [], []
        for bid in battery_ids:
            if bid == test_battery_id:
                continue
            x, y = self._parser_xlsx(df_dict[sheet_names[bid - 1]])
            train_x_list.append(x)
            train_y_list.append(y)

        train_x = np.concatenate(train_x_list, axis=0)
        train_y = np.concatenate(train_y_list, axis=0)

        loaders = self._encapsulation(train_x, train_y, test_x, test_y)
        return dict(zip(['train', 'valid', 'test'], loaders))
    
    def get_full_data(self, test_battery_id=1):
        """Public API: Load full charge & discharge data"""
        path = os.path.join(self.root, 'full', f'Batch{self.batch}_full.mat')
        loaders = self._get_full_raw_data(path, test_battery_id)
        return dict(zip(['train', 'valid', 'test'], loaders))

    def _load_one_batch_full(self, batch_id, with_batt_index=False):
        """Helper to load all batteries within a single batch"""
        path = os.path.join(self.root, 'full', f'Batch{batch_id}_full.mat')
        mat = loadmat(path)
        battery = mat['battery']
        num_batt = battery.shape[1]

        x_list, y_list, bid_list = [], [], []
        for bid in range(num_batt):
            x_i, y_i = self._parser_full_cycle(battery[0, bid][0])
            x_list.append(x_i)
            y_list.append(y_i)

            if with_batt_index:
                bid_list.append(np.full(shape=(x_i.shape[0],), fill_value=bid + 1, dtype=np.int32))
        
        x_all = np.concatenate(x_list, axis=0).astype(np.float32)
        y_all = np.concatenate(y_list, axis=0).astype(np.float32)

        if with_batt_index:
            return x_all, y_all, np.concatenate(bid_list, axis=0)
        return x_all, y_all
    
    def get_full_data_cross_batch(self, train_batches, test_batch):
        """Public API: Cross-batch scenario dataloader mapping"""
        train_x_list, train_y_list = [], []
        for b in train_batches:
            x_b, y_b = self._load_one_batch_full(b)
            train_x_list.append(x_b)
            train_y_list.append(y_b)
            
        train_x = np.concatenate(train_x_list, axis=0)
        train_y = np.concatenate(train_y_list, axis=0)

        test_x, test_y, test_batt_ids = self._load_one_batch_full(test_batch, with_batt_index=True)

        train_loader, valid_loader, test_loader = self._encapsulation(train_x, train_y, test_x, test_y)

        return {
            'train': train_loader,
            'valid': valid_loader,
            'test': test_loader,
            'test_batt_ids': test_batt_ids,
        }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=2023)
    parser.add_argument('--normalized_type', type=str, default='minmax')
    parser.add_argument('--minmax_range', type=tuple, default=(-1, 1))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--batch', type=int, default=1)
    args = parser.parse_args([])

    dataset = XJTUDataset(args)
    loaders = dataset.get_full_data(test_battery_id=1)

    for split in ['train', 'valid', 'test']:
        dl = loaders[split]
        x, y = next(iter(dl))
        print(f"[{split.capitalize():<5}] batch x shape: {x.shape}, batch y shape: {y.shape}")