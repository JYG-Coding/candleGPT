import pandas as pd
import numpy as np

class KlinePreprocessor:
    def __init__(self, ohlcv_df):
        """
        ohlcv_df: pandas.Dat
        aFrame, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        self.df = ohlcv_df.copy()
        self.cleaned = False

    def clean_data(self):
        """
        清洗数据：去除缺失值、重复值、异常值，按时间排序
        """
        # 去除重复行
        self.df = self.df.drop_duplicates()
        # 按时间排序
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        # 去除含有NaN的行
        self.df = self.df.dropna()
        
        #=============================================================
        # 去除价格为负或为零的行
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            print(col)
            self.df = self.df[self.df[col] > 0]
        # 去除成交量为负的行
        self.df = self.df[self.df['volume'] >= 0]
        self.cleaned = True

    def window_normalize(self, window_size=3):
        """
        滑动窗口切分，并对每个窗口做归一化
        返回：list，每个元素为归一化后的窗口ndarray，shape=(window_size, 5)
        """
        if not self.cleaned:
            self.clean_data()
        #data = self.df[['open', 'high', 'low', 'close', 'volume']].values
        data = self.df[['open', 'high', 'low', 'close']].values
        
        #================================================================================
        windows = []
        lenData = len(data)
        for i in range(lenData - window_size + 1):
            window = data[i:i+window_size]
            # 以第一根K线的open为基准做归一化
            base_open = window[0, 0]
            norm_window = window.copy()
            
            norm_window[:, 0:4] = norm_window[:, 0:4] / base_open  # open, high, low, close
            
            """"
            # volume可以选择对数归一化或标准化，这里简单归一化到[0,1]
            vol_min, vol_max = norm_window[:, 4].min(), norm_window[:, 4].max()
            if vol_max > vol_min:
                norm_window[:, 4] = (norm_window[:, 4] - vol_min) / (vol_max - vol_min)
            else:
                norm_window[:, 4] = 0
            """
                
            #===============================================================================
            windows.append(norm_window)
        return windows

    def get_normalized_windows(self, window_size=3):
        """
        直接获取归一化后的滑动窗口序列
        """
        return self.window_normalize(window_size=window_size)

# 使用示例
if __name__ == "__main__":
    # 假设你已经有一个DataFrame: df
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'extra']
    
    df = pd.read_csv('F:\PythonProject\candleGPT\DataTools\XAUUSDmM3.csv', names=columns)
    processor = KlinePreprocessor(df)
    norm_windows = processor.get_normalized_windows(window_size=3)
    print(norm_windows[0])  # 查看第一个窗口的归一化结果
    
    #====================================================================================
    # 保存为npy文件（推荐，适合后续聚类等处理）
    np.save('F:\PythonProject\candleGPT\DataTools/norm_windows.npy', np.array(norm_windows))

    # 如需保存为csv，每个窗口一行，flatten后保存
    norm_windows_flat = [w.flatten() for w in norm_windows]
    norm_df = pd.DataFrame(norm_windows_flat)
    norm_df.to_csv('F:\PythonProject\candleGPT\DataTools/norm_windows.csv', index=False)