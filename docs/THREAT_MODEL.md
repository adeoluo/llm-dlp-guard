# Threat Model - LLM DLP Guard

## Assets

- Personal data and credentials that users and applications put into prompts, and that models
  echo or generate in answers.
- The vault of each request (token to original value) and its HMAC key.
- The policy (what is blocked, redacted or allowed) and the provider API keys that pass through
  the proxy in `Authorization` headers.
- The availability of the LLM integration that the proxy fronts.

## Trust boundaries

1. **Client to proxy:** prompts are untrusted input. They may contain sensitive data by accident
   or be crafted to slip it past the detectors.
2. **Proxy to provider:** everything sent upstream leaves the organisation; this is the boundary
   the guard exists to protect.
3. **Provider to proxy:** answers are untrusted output. They can repeat sensitive data from the
   context, generate realistic secrets, or be steered by injected instructions to leak.

## Threats

| Threat | Vector | Impact | Mitigation |
|---|---|---|---|
| Sensitive data sent to a provider | User pastes a key, a customer record or a private key into a prompt | Disclosure to a third party, logging in provider systems | Patterns, validators and the tagger on every message part; `redact`, `tokenize` or `block` per policy before forwarding |
| Secret split across stream chunks | A credential arrives over several SSE events | Partial or full leak in a streamed answer | Hold-back window and opener rule; cuts never fall inside a candidate; equality with whole-text redaction is property tested (ADR-0006) |
| Evasion by formatting | Spaces, zero-width characters, homoglyphs or encodings (base64, URL encoding) around a value | Detector miss | Partly unmitigated: known formats with separators are matched, encoded or obfuscated values are not; listed under limitations |
| Vault token restoration abuse | A prompt includes text that looks like a token to get it "restored" | Disclosure of another value | Vaults are per request and only contain values from that request's own prompt; restoration needs an exact whole-token match |
| Re-detection loops | Tokenized values look like real cards or IBANs | Double redaction, broken answers | Findings that are vault tokens are skipped on the response path |
| Logging of sensitive data | Access logs or error messages carry bodies | Secondary leak | The service logs nothing about requests; errors return labels and actions, never values |
| Denial of service | Very large bodies, pathological regex inputs | Proxy stalls | 4 MB body cap; Rust's `regex` runs in linear time; patterns are bounded; Python reference is not used in the service by default |
| Policy misconfiguration | An allow rule or allowlist pattern that is too broad | Silent data loss | Rules are explicit and ordered; unknown fields and invalid patterns are rejected when the policy loads |
| Provider credential exposure | The proxy forwards `Authorization` | Key theft if the proxy is exposed | Binds to 127.0.0.1 by default; no authentication of its own, so deploy behind an authenticating gateway |
| Supply chain | Malicious dependency | Code execution | Dependencies pinned in `uv.lock`; CodeQL for Python and Rust in CI |

## Residual risk

Detection is probabilistic. The tagger misses names and addresses it has not seen in similar
contexts (recall 0.84 on ai4privacy, much lower on Nemotron-PII documents), and no pattern covers
proprietary token formats or encoded secrets. The guard reduces what reaches a provider; it does
not replace data minimisation, provider data processing agreements, or keeping credentials out
of prompts altogether.
