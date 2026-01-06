import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, sigma=1.0):
        super().__init__()
        self.temperature = temperature
        self.sigma = sigma

    # SimCLR inspired
    def forward(self, embeddings, survival_times, censor):
        device = embeddings.device
        idx = (censor == 1).nonzero(as_tuple=True)[0]
        if len(idx) < 2:
            return torch.tensor(0.0, requires_grad=True, device=device)

        z = F.normalize(embeddings[idx], dim=1)
        t = survival_times[idx].float()

        sim = z @ z.T / self.temperature
        mask = torch.eye(len(z), device=device).bool()
        sim = sim.masked_fill(mask, -float("inf"))

        diff = (t.unsqueeze(1) - t.unsqueeze(0)).abs()
        std = t.std().clamp(min=1e-6)
        w = torch.exp(-0.5 * (diff / (self.sigma * std)) ** 2)
        w = w.masked_fill(mask, 0)
        w = w / w.sum(1, keepdim=True).clamp(min=1e-8)

        numerator = torch.logsumexp(sim + w.log(), dim=1)
        denominator = torch.logsumexp(sim, dim=1)
        return (-numerator + denominator).mean()


class SimilarityLoss(nn.Module):
    def __init__(self):
        super(SimilarityLoss, self).__init__()

    def forward(self, wsi_embeddings, omic_embeddings):
        """
        want to force embeddings to be similar for the same patient
        """
        # normalize vectors
        cosine_sim = F.cosine_similarity(wsi_embeddings, omic_embeddings)
        cosine_sim = cosine_sim.clamp(min=-1, max=1)

        loss = torch.mean(1 - cosine_sim)

        return loss


class CoxLoss(nn.Module):
    def __init__(self):
        super(CoxLoss, self).__init__()

    def forward(self, log_risks, times, censor):
        """
        :param log_risks: predictions from the NN
        :param times: observed survival times (i.e. times to death) for the batch
        :param censor: censor data, event (death) indicators (1/0)
        :return: Cox loss (scalar)
        """
        device = log_risks.device
        # numerical stability
        log_risks = torch.clamp(log_risks, min=-10, max=10)
        sorted_times, sorted_indices = torch.sort(times, descending=True)
        sorted_log_risks = log_risks[sorted_indices]
        sorted_censor = censor[sorted_indices]

        # precompute for using within the inner sum of term 2 in Cox loss
        exp_sorted_log_risks = torch.exp(sorted_log_risks)

        # initialize all samples to be at-risk (will update it below)
        at_risk_mask = torch.ones_like(sorted_times, dtype=torch.bool)

        losses = []
        for time_index in range(len(sorted_times)):
            # include only the uncensored samples (i.e., for whom the event has happened)
            if sorted_censor[time_index] == 1:
                at_risk_mask = (
                    torch.arange(len(sorted_times)) <= time_index
                )  # less than, as sorted_times is in descending order
                at_risk_mask = at_risk_mask.to(device)
                at_risk_sum = torch.sum(
                    exp_sorted_log_risks[at_risk_mask]  # 2nd term on the RHS
                )  # all are at-risk for the first sample (after arranged in descending order)
                loss = sorted_log_risks[time_index] - torch.log(at_risk_sum + 1e-15)
                losses.append(loss)

            # at_risk_mask[time_index] = False # the i'th sample is no more in the risk-set as the event has already occurred for it

        # if no uncensored samples are in the mini-batch return 0
        if not losses:
            return torch.tensor(0.0, requires_grad=True)

        cox_loss = -torch.mean(torch.stack(losses))
        return cox_loss


class JointLoss(nn.Module):
    def __init__(self, sim_weight=1.0, contrast_weight=0.1, temperature=0.1, sigma=1):
        super(JointLoss, self).__init__()

        self.cox_loss_fn = CoxLoss()
        self.sim_loss_fn = SimilarityLoss()
        # NOTE: performance of the contrastive loss is quite poor
        # self.contrast_loss_fn = ContrastiveLoss(temperature=temperature, sigma=sigma)

        self.sim_weight = sim_weight
        # self.contrast_weight = 0  # contrast_weight

    def forward(self, log_risks, times, censor, wsi_embeddings, omic_embeddings):
        cox_loss = self.cox_loss_fn(log_risks, times, censor)
        sim_loss = self.sim_loss_fn(wsi_embeddings, omic_embeddings)

        # wsi_contrastive_loss = self.contrast_loss_fn(wsi_embeddings, times, censor)
        # omic_contrastive_loss = self.contrast_loss_fn(omic_embeddings, times, censor)
        # contrastive_loss = (wsi_contrastive_loss + omic_contrastive_loss) / 2

        return (
            cox_loss
            + self.sim_weight * sim_loss
            # + self.contrast_weight * contrastive_loss
        )
