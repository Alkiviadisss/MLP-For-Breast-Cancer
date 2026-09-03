import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("breast_cancer_wisconsin.csv")
X = df.drop("target", axis=1)
y = df["target"]

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_temp, y_train, y_temp = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp,test_size=0.50,random_state=42,stratify=y_temp)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val.to_numpy(), dtype=torch.long)
y_test = torch.tensor(y_test.to_numpy(), dtype=torch.long)

class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(30, 64),

            nn.ReLU(),

            nn.Linear(64, 32),

            nn.ReLU(),

            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)


model = MLP()
print(model)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(),lr=0.001)

epochs = 100

for epoch in range(epochs):
    model.train()
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val)
        val_loss = criterion(val_predictions, y_val)
        val_predicted_classes = torch.argmax(val_predictions, dim=1)
        val_accuracy = (val_predicted_classes == y_val).float().mean()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] "f"Train Loss: {loss.item():.4f} "f"Val Loss: {val_loss.item():.4f} "f"Val Accuracy: {val_accuracy.item():.4f}")


model.eval()
with torch.no_grad():
    test_predictions = model(X_test)
    test_predicted_classes = torch.argmax(test_predictions, dim=1)
    test_accuracy = (test_predicted_classes == y_test).float().mean()

print("\nFinal Test Accuracy:", test_accuracy.item())
