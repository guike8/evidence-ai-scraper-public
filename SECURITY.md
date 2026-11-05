# Security and private data

Treat retrieved pages and model output as untrusted input. Keep credentials, SSH identities, model weights, runtime configuration and retrieved private content outside source control. Runtime snapshots and descriptors contain source material; choose a retention and backup policy appropriate to that material.

The direct HTTP adapter validates public HTTP(S) destinations, standard ports and redirects, resolves approved public IPs, and pins native connections while preserving TLS hostname checks. These application checks do not replace deployment network controls. Browser, authenticated and document-processing adapters need their own equivalent boundary checks before integration.

The local inference endpoint is restricted to loopback. Models receive no extraction tools. JSON schema, business completeness and exact-source evidence are checked locally, with a bounded repair attempt. These checks reject unsupported values but do not prove every semantic relationship or eliminate prompt-injection risk on unseen tasks.

Do not include API keys, authorization headers, private snapshots or internal host details in public issues. If the repository offers private vulnerability reporting, use that channel for sensitive reports. Otherwise request a private contact without posting exploit details or sensitive data publicly. Minimal synthetic reproductions are preferred.
