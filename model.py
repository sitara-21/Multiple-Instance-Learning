import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Attention, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, return_attention=False):
        # x: (batch_size, num_instances, input_dim)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        H = F.relu(self.embedding(x))  # (batch_size, num_instances, hidden_dim)
        A = F.softmax(self.attention(H), dim=1)  # (batch_size, num_instances, 1)
        M = torch.sum(A * H, dim=1)  # (batch_size, hidden_dim)
        Y_prob = self.classifier(M).squeeze(-1)  # (batch_size, 1)
        if return_attention:
            Y_hat = torch.ge(Y_prob, 0.5).float()
            return Y_prob, Y_hat, A
        return Y_prob

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X, return_attention=True)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, attention_scores = self.forward(X, return_attention=True)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, attention_scores

class GatedAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GatedAttention, self).__init__()

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), # matrix V
            nn.Tanh()
        )
        
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), # matrix U
            nn.Sigmoid()
        )
        
        self.attention_w = nn.Linear(hidden_dim, hidden_dim) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )


    def forward(self, x, return_attention=False):
        # x: (batch_size, num_instances, input_dim)
        H = F.relu(self.embedding(x))  # (batch_size, num_instances, hidden_dim)
        A_V = self.attention_V(H)  # KxL
        A_U = self.attention_U(H)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        A = F.softmax(A, dim=1)  # (batch_size, num_instances, 1)
        M = torch.sum(A * H, dim=1)  # (batch_size, hidden_dim)
        Y_prob = self.classifier(M).squeeze(-1)  # (batch_size, 1)
        if return_attention:
            Y_hat = torch.ge(Y_prob, 0.5).float()
            return Y_prob, Y_hat, A
        return Y_prob

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A