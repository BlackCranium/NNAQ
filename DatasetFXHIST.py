import pandas as pd
import numpy as np
import torch.utils
from torch.utils.data import Dataset

class DatasetFXHIST(Dataset):
    file_path = 'data/eurusdhist/eurusd_minute.csv'
    use_cols = ['Minute',
                'BidOpen', 'BidHigh', 'BidLow', 'BidClose',
                'BidChange',
                'AskOpen', 'AskHigh', 'AskLow', 'AskClose',
                'AskChange']
    n_rows = None
    paramPrediction = {'tp': 400, 'sl': 200, 'point': 0.00001, 'prediction range':100}

    def __init__(self):
        self.data = pd.read_csv(self.file_path,
                                delimiter=',',
                                usecols=self.use_cols,
                                nrows=self.n_rows
                                )


    def __len__(self):
        return len(self.data)

    @property
    def Minute(self):
        return self.data['Minute'].to_numpy()

    @property
    def BidOpen(self):
        return self.data['BidOpen'].to_numpy()

    @property
    def BidHigh(self):
        return self.data['BidHigh'].to_numpy()

    @property
    def BidLow(self):
        return self.data['BidLow'].to_numpy()

    @property
    def BidClose(self):
        return self.data['BidClose'].to_numpy()

    @property
    def BidChange(self):
        return self.data['BidChange'].to_numpy()

    @property
    def AskOpen(self):
        return self.data['AskOpen'].to_numpy()

    @property
    def AskHigh(self):
        return self.data['AskHigh'].to_numpy()

    @property
    def AskLow(self):
        return self.data['AskLow'].to_numpy()

    @property
    def AskClose(self):
        return self.data['AskClose'].to_numpy()

    @property
    def AskChange(self):
        return self.data['AskChange'].to_numpy()


    def __getitem__(self, index):
        """
    Args:
        index (int): Index

    Returns:
        tuple: (quotation, prediction) where prediction is index of the prediction class.
    """
        start,stop=int(max(1, index - 60)),int(max(1, index))
        # l = self.data.values[int(max(1, index - 60)):index, 1:]
        a=np.block([self.Minute[start:stop],
                    self.BidOpen[start:stop],
                    self.BidOpen[start:stop],
                    self.BidHigh[start:stop],
                    self.BidLow[start:stop],
                    self.BidClose[start:stop],
                    self.BidChange[start:stop],
                    self.AskOpen[start:stop],
                    self.AskHigh[start:stop],
                    self.AskLow[start:stop],
                    self.AskClose[start:stop],
                    self.AskChange[start:stop]]
                   )

        quotation = a#torch.tensor(a)
        pred = self.getPrediction(index)
        # prediction=torch.as_tensor([0,0,0])
        # prediction[pred+1]=1
        prediction =pred
        return quotation, prediction

    def getPrediction(self, index: int):
        tp, sl, pnt, range = self.paramPrediction.values()

        start, stop = index, index + range
        ao = self.AskOpen[start:stop]
        bh = self.BidHigh[start:stop]
        bl = self.BidLow[start:stop]
        bo = self.BidOpen[start:stop]
        al = self.AskLow[start:stop]
        ah = self.AskHigh[start:stop]

        pr_lose = {'buy': bo[0] - sl * pnt,
                   'sell': ao[0] + sl * pnt}
        pr_profit = {'buy': ao[0] + tp * pnt,
                     'sell': bo[0] - tp * pnt}
        isProf = {'buy': False, 'sell': False}
        isLoss = {'buy': False, 'sell': False}
        #for i in range(start, stop-1):
        for i, _ in enumerate(bh):
            isProf['buy'] = bh[i] >= pr_profit['buy']
            isProf['sell'] = al[i] <= pr_profit['sell']
            isLoss['buy'] = isLoss['buy'] or bl[i] <= pr_lose['buy']
            isLoss['sell'] = isLoss['sell'] or ah[i] >= pr_lose['sell']
            # print(i,'LOSE',isLoss,'PROF',isProf)
            if isLoss['buy'] and isLoss['sell']:
                # print(f'{start}-{i},LOSS [{(pr_lose["buy"]-ao[start])/pnt}] [{(-pr_lose["sell"]+bo[start])/pnt}],({isLoss})')
                return 0
            if isProf['buy']:
                # print(f'{start}-{i},BUY PROF [{(pr_profit["buy"]-ao[start])/pnt}],({isProf},{isLoss})')
                return 1
            if isProf['sell']:
                # print(f'{start}-{i},SEL PROF [{(-pr_profit["sell"]+bo[start])/pnt}],({isProf},{isLoss})')
                return -1
        return 0

    def __getDate__(self, index):
        return self.data.iloc[index, 0]

    def getYear(self, index):
        return int(self.__getDate__(index)[:4])

    def getMonth(self, index):
        return int(self.__getDate__(index)[5:7])

    def getDay(self, index):
        return int(self.__getDate__(index)[-2:])

    def getHour(self, index):
        try:
            return int(self.Minute[index] // 60)
        except:
            try:
                return int(self.data.Hour[index])
            except:
                return int(self.data.Hour[index][:2])



    def getMinute(self, index):
        try:
            return int(self.data.Minute[index])
        except:
            try:
                return int(self.data.Hour[index]) * 60
            except:
                return int(self.data.Hour[index][:2]) * 60


class FXHISTDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        super(FXHISTDataLoader, self).__init__(dataset, batch_size, shuffle)
