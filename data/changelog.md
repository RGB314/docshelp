# Nimbus Notes changelog

## 4.12 (released 2026-10-06)
- Sync: a new delta-sync engine uploads only changed paragraphs. Large notebooks sync up to 5x faster.
- Known issue: on Windows, the tray icon can show a red slash for a few minutes after waking from sleep, even
  though sync works. A fix is planned for 4.12.1.
- Export: PDF export keeps internal links between notes.

## 4.11 (released 2026-09-15)
- Sharing: share a notebook as read-only with people outside your workspace (Team and Enterprise).
- Security: SSO now supports SCIM provisioning for Okta and Entra ID (Enterprise).
- Fixed: the importer skipped attachments larger than 25 MB without a warning; it now lists them.

## 4.10 (released 2026-08-25)
- Offline: the mobile apps keep the 200 most recent notes available offline (previously 50).
- API: new `/v2/notes/search` endpoint with full-text and tag filters.
- Fixed: trash retention was shown as 7 days in the iOS app; it is 30 days on every platform.
