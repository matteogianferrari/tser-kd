import torch
import torch.nn as nn


class TSCELoss(nn.Module):
    """Temporal Separation Cross Entropy loss function.


    """

    def __init__(self, tau: float = 1.0) -> None:
        """

        Args:
            tau:
        """
        super(TSCELoss, self).__init__()

        self.tau = tau

        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, stu_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """

        Args:
            stu_logits:
            targets:

        Returns:

        """
        # Retrieves the shape of the student logits
        T, B, K = stu_logits.shape

        # Scales the logits by the temperature
        stu_logits = stu_logits / self.tau

        # Vectorized the tensor to perform CE without a loop
        flattened = stu_logits.view(T * B, K)

        # Repeats the targets to match the vectorized logits
        repeated_targets = targets.unsqueeze(0).repeat(T, 1).view(T * B)

        # Computes the CE loss over each single time step
        loss_val = self.ce(flattened, repeated_targets)

        # Averages over T and over B
        loss_val = loss_val.mean()

        return loss_val


class TSKLLoss(nn.Module):
    """Temporal Separation Kullback–Leibler Divergence loss function.

    """

    def __init__(self, tau: float = 1.0) -> None:
        """

        Args:
            tau:
        """
        super(TSKLLoss, self).__init__()

        self.tau = tau

    def forward(self, tea_logits, stu_spk_out):
        log_p_stu = F.log_softmax(stu_spk_out / self.tau, dim=-1)
        #print(log_p_stu.shape)

        log_p_tea = F.log_softmax(tea_logits / self.tau, dim=-1)
        #print(log_p_tea.shape)
        log_p_tea = log_p_tea.unsqueeze(0).expand_as(log_p_stu)
        #print(log_p_tea.shape)

        #print("Questo")
        # Perform KL divergence over T dimension
        kl_per_t = F.kl_div(input=log_p_stu, target=log_p_tea, log_target=True, reduction='none').sum(-1)
        kl_per_sample = kl_per_t.mean(0)
        #print(kl_per_sample.shape)

        kl = kl_per_sample.mean(0)
        #print(kl.shape)

        # Perform mean over T, output is [B]=[512]
        # tau squared act as output regularizer in knowledge distillation
        return kl * (self.tau ** 2)


class EntropyReg(nn.Module):
    """Entropy Regularization loss function.

    """

    def __init__(self, tau:float=1.0):
        super(EntropyReg, self).__init__()

        self.tau = tau

    def forward(self, spk_out):
        p = F.softmax(spk_out / self.tau, dim=-1)
        log_p = F.log_softmax(spk_out / self.tau, dim=-1)


        return -(p * log_p).sum(-1).mean(0).mean(0)


class TSERKDLoss(nn.Module):
    """Temporal Separation with Entropy Regularization for Knowledge Distillation loss function.

    """

    def __init__(self, alpha: float, gamma: float, tau: float = 1.0) -> None:
        """

        Args:
            alpha:
            gamma:
            tau:
        """
        super(TSERKDLoss, self).__init__()

        self.tau = tau

        self.alpha = alpha
        self.gamma = gamma

        self.ce_loss = TSCELoss()
        self.kl_loss = TSKLLoss()
        self.e_reg = EntropyReg()

    def forward(self, tea_logits, stu_spk_out, targets):
        ce_loss = self.ce_loss(stu_spk_out, targets)
        kl_loss = self.kl_loss( tea_logits=tea_logits, stu_spk_out=stu_spk_out)
        e_reg = self.e_reg(stu_spk_out)

        total_loss = (1 - self.alpha) * ce_loss + self.alpha * kl_loss - self.gamma * e_reg

        return total_loss, ce_loss, kl_loss, e_reg
