import torch
import torch.nn as nn
import pickle
import numpy as np

# -----------------------------
# Load Events
# -----------------------------

with open("data/processed/events.pkl", "rb") as f:
    events = pickle.load(f)

print("Total available patients:", len(events))


# -----------------------------
# Model Definition (must match training)
# -----------------------------

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

        return intensity


# -----------------------------
# Load Trained Model
# -----------------------------

model = VariationalHawkes()
model.load_state_dict(torch.load("models/variational_hawkes.pth"))
model.eval()

print("\nModel loaded successfully.\n")


# =====================================================
# Interactive Patient Prediction
# =====================================================

patient_id = input("Enter patient ID: ")

if patient_id not in events:
    print("Patient ID not found.")
    exit()

seq = events[patient_id]

if len(seq) < 2:
    print("Not enough events for this patient.")
    exit()

with torch.no_grad():

    times = torch.tensor([e[0] for e in seq], dtype=torch.float32)
    types = torch.tensor([e[1] for e in seq], dtype=torch.long)

    delta_t = torch.diff(times, prepend=times[0:1])

    times = times.unsqueeze(0)
    types = types.unsqueeze(0)
    delta_t = delta_t.unsqueeze(0)

    intensity = model(types, delta_t, times)

    lambda_last = intensity[0, -1]

    # Next event probabilities
    prob = lambda_last / torch.sum(lambda_last)

    # Expected next event time (Monte Carlo)
    lambda_total = torch.sum(lambda_last)

    sim_samples = []
    for _ in range(100):
        u = torch.rand(1)
        dt_sample = -torch.log(u) / (lambda_total + 1e-8)
        sim_samples.append(dt_sample.item())

    expected_dt = np.mean(sim_samples)

    # Risk score
    risk = lambda_total.item()

print("\n===== Patient-Specific Prediction =====")
print(f"\nPatient ID: {patient_id}")

print("\nNext Event Probabilities:")
print(f"Glucose:     {prob[0].item():.4f}")
print(f"HbA1c:       {prob[1].item():.4f}")
print(f"Creatinine:  {prob[2].item():.4f}")

print("\nExpected Time to Next Event:")
print(f"{expected_dt:.4f} time units")

print("\nRisk Score:")
print(f"{risk:.6f}")

print("\nLearned Alpha Matrix:")
print(model.softplus(model.alpha).detach().numpy())

print("\nLearned Beta:")
print(model.softplus(model.beta).item())