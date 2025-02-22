import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, in_features=16, h1=64, h2=28, out_features=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

def graph(epochs: int, losses: list[float]) -> None:
    plt.plot(range(epochs), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss_plot.png')
    plt.show()

torch.manual_seed(392)

model = Model()

dataframe = pd.read_csv(r'stroke-risk-prediction-dataset\stroke_risk_dataset.csv')

X = dataframe.drop('Stroke Risk (%)', axis=1).drop('At Risk (Binary)', axis=1).values
y = dataframe['At Risk (Binary)'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=392)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

epochs = 1000
losses = []
print('Training...')
for i in range(1, epochs + 1):
    y_pred = model.forward(X_train)

    loss = criterion(y_pred, y_train)
    losses.append(loss.item())

    if i % 10 == 0:
        print(f'Epoch: {i} Loss: {loss}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('Training finished!\n')

graph(epochs, losses)

print(f'Testing...')
with torch.no_grad():
    for i, y in enumerate(X_test):
        y_val = model.forward(y)

        print(f'Test {i+1:3d} | Prediction: {y_val.tolist()} | Actual: {y_test[i]} | Class: {y_val.argmax().item()}')

with torch.no_grad():
    predictions = model(X_test).argmax(dim=1)
correct = (predictions == y_test).sum().item()
print(f'{correct}/{len(y_test)} correct! ({accuracy_score(y_test, predictions) * 100:.2f}%)')

torch.save(model.state_dict(), 'stroke_risk_classifier_model.pth')