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

## Standards mapping

Each threat is mapped to the OWASP Top 10 for LLM Applications 2026 and MITRE ATLAS v2026.10
(IDs and names checked against `ATLAS.yaml` from the
[v2026.10 release of mitre-atlas/atlas-data](https://github.com/mitre-atlas/atlas-data/releases)).
The OWASP Top 10 for Agentic Applications 2026 is not mapped: the guard inspects model traffic
and has no view of agent tools, memory or identity.

| Threat | OWASP LLM 2026 | MITRE ATLAS v2026.10 | Role of this project |
|---|---|---|---|
| Sensitive data sent to a provider | LLM02:2026 Sensitive Information Disclosure | AML.T0085.002 Data from AI Services: Conversation Context and Memory; AML.T0055 Unsecured Credentials | Mitigates: detection, then redaction, tokenization or blocking before forwarding |
| Answers that repeat or generate sensitive data | LLM02:2026 Sensitive Information Disclosure | AML.T0136 Exfiltration via AI User Interface; AML.T0137.003 Training Data Information from AI Models: Extract Memorized Data | Mitigates: the same detectors run on the response path |
| Secret split across stream chunks | LLM02:2026 Sensitive Information Disclosure | AML.T0136 Exfiltration via AI User Interface | Mitigates: hold-back window, property tested against whole-text redaction |
| Injected instructions that steer the model to leak context | LLM01:2026 Prompt Injection; LLM02:2026 Sensitive Information Disclosure | AML.T0051.001 LLM Prompt Injection: Indirect | Partly: leaked values are redacted on the way out; the injection itself is not detected |
| Evasion by formatting or encoding | LLM02:2026 Sensitive Information Disclosure | AML.T0123 Obfuscated Files or Information | Partly: separators are handled, encoded values are not |
| Vault token restoration abuse | LLM01:2026 Prompt Injection; LLM02:2026 Sensitive Information Disclosure | AML.T0051.000 LLM Prompt Injection: Direct | Mitigates: per-request vaults and exact whole-token restoration |
| Logging of sensitive data | LLM02:2026 Sensitive Information Disclosure | AML.T0055 Unsecured Credentials | Mitigates: the service logs no request content |
| Denial of service | LLM06:2026 Unbounded Consumption | AML.T0029 Denial of AI Service | Mitigates with a body cap and linear-time regex (AML.M0036 Limit AI Workload Resource Consumption) |
| Provider credential exposure | LLM02:2026 Sensitive Information Disclosure | AML.T0055 Unsecured Credentials; AML.T0132 Misconfigured or Publicly Exposed AI Services | Partly: binds to 127.0.0.1; authentication is left to a gateway in front |
| Malicious dependency | LLM04:2026 Supply Chain | AML.T0010.001 Supply Chain Compromise: Software | Mitigates with a lockfile and CodeQL (AML.M0016 Vulnerability Scanning) |

The guard as a whole is an instance of AML.M0020 Generative AI Guardrails on both directions of
traffic. Policy misconfiguration has no ATLAS technique; it is covered by strict policy loading.
In NIST AI 600-1 (Generative AI Profile) terms the project addresses the Data Privacy and
Information Security risks.

## Residual risk

Detection is probabilistic. The tagger misses names and addresses it has not seen in similar
contexts (recall 0.84 on ai4privacy, much lower on Nemotron-PII documents), and no pattern covers
proprietary token formats or encoded secrets. The guard reduces what reaches a provider; it does
not replace data minimisation, provider data processing agreements, or keeping credentials out
of prompts altogether.
