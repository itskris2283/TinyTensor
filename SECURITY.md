# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 0.1.x | Yes |

## Reporting a Vulnerability

If you discover a security vulnerability or integer overflow exploit in TinyTensor:
1. Please do not open a public issue.
2. Email the maintainers directly with reproduction steps and target environment details.
3. Vulnerabilities will be triaged and addressed with an emergency patch release within 48 hours.

## Security Architecture

TinyTensor implements multi-layered security controls:
- **Checked Arithmetic:** All dimension multiplication and allocation arithmetic uses `tinytensor::checked_multiply` to prevent integer overflow and heap exploitation.
- **Resource Limits:** Maximum tensor rank is bounded to 8, and maximum individual allocations are capped at 4 GB.
- **Sandboxed Compilation:** The C++ code generator invokes compilers via direct POSIX `fork()` and `execve()` system calls with discrete argument arrays, strictly eliminating shell command injection risks.
- **Bounded Subprocesses:** Compilation runs in dedicated process groups with bounded timeouts to prevent resource exhaustion attacks.
