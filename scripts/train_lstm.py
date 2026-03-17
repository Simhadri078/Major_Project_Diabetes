import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# -----------------------------
# 1️⃣ Load sequences
# -----------------------------

X = np.load("data/processed/X_sequences.npy")
y = np.load("data/processed/y_targets.npy")

print("Loaded X:", X.shape)
print("Loaded y:", y.shape)

# -----------------------------
# 2️⃣ Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# -----------------------------
# 3️⃣ Define LSTM Model
# -----------------------------

class VAELSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, latent_dim=64, output_size=3):
        super(VAELSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)
        
        # Decoder
        self.fc_out = nn.Linear(latent_dim, output_size)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        
        z = self.reparameterize(mu, logvar)
        
        output = self.fc_out(z)
        
        return output, mu, logvar

model = VAELSTM()

# -----------------------------
# 4️⃣ Loss and Optimizer
# -----------------------------

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# -----------------------------
# 5️⃣ Training Loop
# -----------------------------

epochs = 30

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for xb, yb in train_loader:
        optimizer.zero_grad()
        predictions, mu, logvar = model(xb)

# Reconstruction loss
        recon_loss = criterion(predictions, yb)

# KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

# Total loss (beta can be tuned)
        beta_kl = 0.0001
        loss = recon_loss + beta_kl * kl_loss
        loss.backward()

# Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# -----------------------------
# 6️⃣ Evaluation
# -----------------------------

from sklearn.metrics import mean_absolute_error, r2_score

model.eval()
with torch.no_grad():
    predictions, _, _ = model(X_test)
    test_loss = criterion(predictions, y_test)

    pred = predictions.numpy()
    true = y_test.numpy()

print("Test MSE:", test_loss.item())
print("MAE:", mean_absolute_error(true, pred))
print("R2 Score:", r2_score(true, pred))