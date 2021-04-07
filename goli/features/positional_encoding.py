from goli.features.spectral import compute_laplacian_eigenfunctions


def graph_positional_encoder(adj, pos_type: str, num_pos: int, disconnect: bool = True, **kwargs):

    pos_type = pos_type.lower()

    if pos_type == "laplacian_eigenvectors":
        pos_enc = compute_laplacian_eigenfunctions(
            adj=adj, num_pos=num_pos, disconnect=disconnect, **kwargs
        )
    else:
        raise ValueError(f"Unknown `pos_type`: {pos_type}")

    return pos_enc

