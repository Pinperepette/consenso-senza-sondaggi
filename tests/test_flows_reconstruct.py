"""La matrice di transizione stimata deve ricostruire i marginali dell'elezione B
a partire dalle quote di A (vincolo contabile q = P^T x)."""
import numpy as np

from consenso.model.flows import FlowData, flow_model, summarize_flows


def test_flow_reconstructs_marginals():
    import jax
    from numpyro.infer import MCMC, NUTS

    rng = np.random.default_rng(0)
    cats = ["party:A", "party:B", "astensione"]
    # matrice di transizione vera (righe = da, sommano a 1), diagonale dominante
    P_true = np.array([[0.75, 0.10, 0.15],
                       [0.08, 0.80, 0.12],
                       [0.20, 0.15, 0.65]])
    A = 40
    X = rng.dirichlet(np.array([3.0, 3.0, 2.0]), size=A)     # quote A per area
    N = rng.integers(800, 1500, size=A)
    q_true = np.einsum("ai,ij->aj", X, P_true)
    CB = np.array([rng.multinomial(N[a], q_true[a]) for a in range(A)])

    data = FlowData(categories_from=cats, categories_to=cats, x=X,
                    counts_b=CB, N=N, hierarchical=False)
    mcmc = MCMC(NUTS(flow_model), num_warmup=300, num_samples=300,
                num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(1), data)
    post = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}

    Pbar = post["Pbar"].mean(0)
    # ricostruzione dei marginali B
    q_pred = np.einsum("ai,ij->aj", X, Pbar)
    observed_b = CB / CB.sum(axis=1, keepdims=True)
    recon_err = np.mean(np.abs(q_pred - observed_b))
    assert recon_err < 0.03, f"errore ricostruzione marginali troppo alto: {recon_err}"

    # la fedeltà (diagonale) deve essere recuperata in modo sensato
    summary = summarize_flows(post, data)
    assert summary["loyalty"]["party:A"] > 0.55
    assert summary["loyalty"]["party:B"] > 0.6
