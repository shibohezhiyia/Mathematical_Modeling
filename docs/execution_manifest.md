# Execution manifest

`build_execution_manifest` records run identity, input digest, seed, dependency
versions, generated-code digest and log digest. It intentionally stores hashes,
not raw code, user data or logs, so artifact stores can link reproducibility
without widening the data-exposure boundary.
