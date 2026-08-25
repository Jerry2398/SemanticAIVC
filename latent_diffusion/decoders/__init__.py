"""Decoder-adapter registry. get_decoder('scvi') -> SCVIDecoder()."""


def get_decoder(name, **kwargs):
    if name == "scvi":
        from decoders.scvi_decoder import SCVIDecoder
        return SCVIDecoder(**kwargs)
    if name == "scgen":
        from decoders.scgen_decoder import SCGENDecoder
        return SCGENDecoder(**kwargs)
    if name == "trvae":
        from decoders.trvae_decoder import TRVAEDecoder
        return TRVAEDecoder(**kwargs)
    if name.startswith("tahoe_vae"):
        from decoders.tahoe_vae_decoder import TahoeVAEDecoder
        return TahoeVAEDecoder(name=name, **kwargs)   # tahoe_vae / tahoe_vae_nb / tahoe_vae_scgen
    raise ValueError(f"no decoder adapter for {name!r} "
                     "(available: scvi, scgen, trvae, tahoe_vae*; scDisInFact/Squidiff not natural fits)")
