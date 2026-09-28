# Security

## Encryption

All data is encrypted in transit with TLS 1.3 and at rest with AES-256. Enterprise workspaces can bring their own
encryption keys (BYOK) through AWS KMS.

## Two-factor authentication

Any user can turn on two-factor authentication (2FA) with an authenticator app under **Profile → Security**.
Admins on Team and Enterprise plans can require 2FA for all members.

## Single sign-on

SAML single sign-on (Okta, Azure AD / Entra ID, Google Workspace) and SCIM user provisioning are available on the
Enterprise plan only.

## Data residency

Workspaces are hosted in the US by default. Enterprise customers can choose EU (Frankfurt) hosting.

## Compliance

Nimbus Notes is SOC 2 Type II certified and GDPR compliant.
