from torch.utils.data import DataLoader, Dataset

class ExperienceBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, state, action, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def __len__(self): 
        return len(self.states)

class GymDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# class MyDataModule(pl.LightningDataModule):
#     def __init__(self, data):
#         super().__init__()
#         self.data = data

#     def train_dataloader(self):
#         return DataLoader(GymDataset(self.data), batch_size=32, shuffle=True)
