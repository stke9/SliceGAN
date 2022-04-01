import torch

noise_distributions = {
    "normal" : torch.distributions.normal.Normal(0,1),
    "laplace" : torch.distributions.laplace.Laplace(0,1),
    "uniform" : torch.distributions.uniform.Uniform(-1,1),
    "cauchy": torch.distributions.cauchy.Cauchy(0,1)

}

k = noise_distributions["uniform"]

t = k.sample((2,2,2))

print(t)