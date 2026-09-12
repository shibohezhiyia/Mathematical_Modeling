# Failure diagnosis

`diagnose_failure` checks numerical tolerance, observation error, local
identifiability and structural misspecification in that order. It preserves
alternative explanations and explicitly refuses to infer a latent state merely
because a residual is large.
