# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

1. **DO NOT** create a public GitHub issue for security vulnerabilities
2. Email the maintainers directly or use GitHub's private vulnerability reporting feature
3. Include as much detail as possible:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: We will acknowledge receipt within 48 hours
- **Assessment**: We will assess the vulnerability within 7 days
- **Resolution**: We aim to release a fix within 30 days for confirmed vulnerabilities
- **Credit**: We will credit reporters in the release notes (unless you prefer anonymity)

### Scope

This security policy covers:
- The motcpp library code
- Build system vulnerabilities
- Dependencies with known CVEs

### Out of Scope

- Vulnerabilities in upstream dependencies (report to respective projects)
- Issues requiring physical access to the machine
- Social engineering attacks

## Security Best Practices

When using motcpp:

1. **Keep Updated**: Always use the latest version
2. **Validate Input**: Sanitize detection inputs before passing to trackers
3. **Model Security**: Only load ONNX models from trusted sources
4. **Build Security**: Use release builds with appropriate compiler flags

## Security Features

motcpp includes:
- No network access (air-gapped operation possible)
- No dynamic code execution
- Bounded memory allocation
- Input validation for detection matrices

## Contact

For security concerns, contact the maintainers through GitHub.
