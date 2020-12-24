import torch
print(torch.__version__)
import torchvision
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from tools.util import get_model
import tools.defaults as defaults
import tools.data_util as data_util

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class ColorDataset(Dataset):
    def __init__(self, is_train):
        self._colors = defaults.COLORS
        if is_train:
            self.label_names = [random.choice(self._colors) for i in range(1000)]
        else:
            self.label_names = self._colors.copy()
        self.transforms = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.label_names)

    def __getitem__(self, idx):
        label_name = self.label_names[idx]
        label = self._colors.index(label_name)
        image = data_util.get_image(color=label_name)
        image = self.transforms(image)
        sample = {"image": image, "label": label}
        return sample


def train(torch_model_path):
    """
    Classify 5 color.
    This model is good for trying serving because you don't need real images.
    """
    print("Start Training.")
    model = get_model()
    optimizer =  torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    data_loader = DataLoader(ColorDataset(is_train=True), batch_size=64, shuffle=True, num_workers=4)
    for epoch in range(10):
        model.train()
        train_pbar = tqdm(data_loader, total=len(data_loader))
        for sample in train_pbar:
            images, labels = sample['image'].type(torch.FloatTensor).to(DEVICE), \
                             sample['label'].to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(input=logits, target=labels)
            loss.backward()
            optimizer.step()
            train_pbar.set_postfix(loss=loss.data.cpu().numpy(), epoch=epoch)

    torch.save(model.state_dict(), torch_model_path)
    print("Finished Training.")
    print("Torch Model has saved to {}.".format(torch_model_path))


def test(model_path):
    print("Start Testing.")
    model = get_model(pretrained_model_path=model_path)
    data_loader = DataLoader(ColorDataset(is_train=False), batch_size=8, shuffle=False)
    model.eval()

    y_true = []
    y_pred = []
    for sample in data_loader:
        images, labels = sample['image'].type(torch.FloatTensor).to(DEVICE), \
                         sample['label'].to(DEVICE)

        logits = model(images)
        y_true += labels.cpu().tolist()
        y_pred += torch.argmax(logits, dim=1).cpu().tolist()

    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print("Accuracy = {}.".format(accuracy))
    assert accuracy == 1.0


if __name__ == '__main__':
    train(defaults.TORCH_MODEL_PATH)
    test(defaults.TORCH_MODEL_PATH)