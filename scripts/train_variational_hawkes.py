import torch
import torch.nn as nn
import pickle
import numpy as np
import os

# =============================
# Load event data
# =============================

with open("data/processed/events.pkl", "rb") as f:
    events = pickle.load(f)

all_sequences = []
for patient_id, seq in events.items():
    if len(seq) >= 2:
        all_sequences.append(seq)

print("Total sequences:", len(all_sequences))

# -----------------------------
# Train / Test Split
# -----------------------------

split_idx = int(0.8 * len(all_sequences))
train_sequences = all_sequences[:split_idx]
test_sequences = all_sequences[split_idx:]

print("Train sequences:", len(train_sequences))
print("Test sequences:", len(test_sequences))

# =============================
# Variational Multivariate Hawkes Model
# =============================

class VariationalHawkes(nn.Module):
    def __init__(self, num_types=3, hidden_size=32, latent_dim=16):
        super().__init__()

        self.num_types = num_types

        self.embedding = nn.Embedding(num_types, 8)
        self.gru = nn.GRU(9, hidden_size, batch_first=True)

        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)
        self.fc_intensity = nn.Linear(latent_dim, num_types)

        self.alpha = nn.Parameter(torch.randn(num_types, num_types) * 0.1)
        self.beta = nn.Parameter(torch.tensor(0.1))

        self.softplus = nn.Softplus()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, event_types, delta_t, times):

        emb = self.embedding(event_types)
        x = torch.cat([emb, delta_t.unsqueeze(-1)], dim=-1)

        h, _ = self.gru(x)

        mu_latent = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu_latent, logvar)

        neural_part = self.fc_intensity(z)

        beta_pos = self.softplus(self.beta)
        alpha_pos = self.softplus(self.alpha)

        batch_size, seq_len = event_types.shape
        excitation = torch.zeros_like(neural_part)

        for i in range(seq_len):
            for j in range(i):
                dt = times[:, i] - times[:, j]
                decay = torch.exp(-beta_pos * dt)
                type_j = event_types[:, j]

                for k in range(self.num_types):
                    excitation[:, i, k] += alpha_pos[k, type_j] * decay

        intensity = self.softplus(neural_part + excitation)

        return intensity, mu_latent, logvar


# =============================
# Training Setup
# =============================

model = VariationalHawkes()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

epochs = 30
beta_kl = 0.001
lambda_reg_weight = 1e-4  # NEW: intensity regularization

# =============================
# Training Loop
# =============================

for epoch in range(epochs):

    total_loss = 0
    total_events = 0

    for seq in train_sequences:

        times = torch.tensor([e[0] for e in seq], dtype=torch.float32)
        types = torch.tensor([e[1] for e in seq], dtype=torch.long)

        delta_t = torch.diff(times, prepend=times[0:1])

        times = times.unsqueeze(0)
        types = types.unsqueeze(0)
        delta_t = delta_t.unsqueeze(0)

        intensity, mu, logvar = model(types, delta_t, times)

        # -----------------------------
        # Log-likelihood
        # -----------------------------

        lambda_selected = intensity[0, torch.arange(len(types[0])), types[0]]
        log_likelihood = torch.sum(torch.log(lambda_selected + 1e-8))

        lambda_sum = torch.sum(intensity, dim=-1)

        # -----------------------------
        # 🔥 FIX 1: Trapezoidal Integral
        # -----------------------------

        lambda_shifted = torch.cat(
            [lambda_sum[:, :1], lambda_sum[:, :-1]], dim=1
        )

        integral = torch.sum(
            0.5 * (lambda_sum + lambda_shifted) * delta_t
        )

        recon_loss = -(log_likelihood - integral)

        # -----------------------------
        # KL divergence
        # -----------------------------

        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        # -----------------------------
        # 🔥 FIX 2: Intensity Regularization
        # -----------------------------

        lambda_reg = lambda_reg_weight * torch.mean(intensity ** 2)

        loss = recon_loss + beta_kl * kl_loss + lambda_reg

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        total_events += len(seq)

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    print("Avg loss per event:", total_loss / total_events)
    print("Beta:", model.softplus(model.beta).item())
    print("Alpha matrix:")
    print(model.softplus(model.alpha).detach().numpy())
    print("-" * 50)

# =============================
# Save Model
# =============================

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/variational_hawkes.pth")

print("Model saved successfully.")