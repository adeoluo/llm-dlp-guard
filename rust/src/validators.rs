//! Validators run on a candidate match. Each returns `None` to reject the match or a factor in
//! (0, 1] that multiplies the pattern's base score.

use std::collections::BTreeMap;
use std::net::Ipv6Addr;

/// Validator names accepted in a catalog.
pub const VALIDATORS: &[&str] = &[
    "luhn",
    "iban",
    "email",
    "ipv4",
    "ipv6",
    "jwt",
    "github_token",
    "ssn",
    "phone",
    "secret_value",
    "high_entropy",
];

/// Checksum and structure validators; their matches win overlaps with tagger spans.
pub const STRUCTURED: &[&str] = &["luhn", "iban", "ipv4", "ipv6", "email"];

/// Run validator `name` on `value`.
pub fn check(name: &str, value: &str) -> Option<f64> {
    match name {
        "luhn" => luhn_valid(value).then_some(1.0),
        "iban" => iban_valid(value).then_some(1.0),
        "email" => email_valid(value).then_some(1.0),
        "ipv4" => ipv4_factor(value),
        "ipv6" => ipv6_factor(value),
        "jwt" => jwt_valid(value).then_some(1.0),
        "github_token" => Some(if github_checksum_valid(value) {
            1.0
        } else {
            0.8
        }),
        "ssn" => ssn_valid(value).then_some(1.0),
        "phone" => phone_factor(value),
        "secret_value" => secret_value_factor(value),
        "high_entropy" => (shannon_entropy(value) >= 3.0).then_some(1.0),
        _ => None,
    }
}

/// Luhn checksum over the digits of a card-like string (spaces and dashes allowed).
pub fn luhn_valid(value: &str) -> bool {
    let mut digits: Vec<u32> = Vec::with_capacity(value.len());
    for c in value.chars() {
        match c {
            '0'..='9' => digits.push(c as u32 - '0' as u32),
            ' ' | '-' => {}
            _ => return false,
        }
    }
    if !(12..=19).contains(&digits.len()) || digits.iter().all(|&d| d == digits[0]) {
        return false;
    }
    let mut sum = 0;
    for (i, &d) in digits.iter().rev().enumerate() {
        if i % 2 == 1 {
            let x = d * 2;
            sum += if x > 9 { x - 9 } else { x };
        } else {
            sum += d;
        }
    }
    sum % 10 == 0
}

/// IBAN lengths by country code (SWIFT IBAN registry).
pub const IBAN_LENGTHS: &[(&str, usize)] = &[
    ("AD", 24),
    ("AE", 23),
    ("AL", 28),
    ("AT", 20),
    ("AZ", 28),
    ("BA", 20),
    ("BE", 16),
    ("BG", 22),
    ("BH", 22),
    ("BI", 27),
    ("BR", 29),
    ("BY", 28),
    ("CH", 21),
    ("CR", 22),
    ("CY", 28),
    ("CZ", 24),
    ("DE", 22),
    ("DJ", 27),
    ("DK", 18),
    ("DO", 28),
    ("EE", 20),
    ("EG", 29),
    ("ES", 24),
    ("FI", 18),
    ("FK", 18),
    ("FO", 18),
    ("FR", 27),
    ("GB", 22),
    ("GE", 22),
    ("GI", 23),
    ("GL", 18),
    ("GR", 27),
    ("GT", 28),
    ("HN", 28),
    ("HR", 21),
    ("HU", 28),
    ("IE", 22),
    ("IL", 23),
    ("IQ", 23),
    ("IS", 26),
    ("IT", 27),
    ("JO", 30),
    ("KW", 30),
    ("KZ", 20),
    ("LB", 28),
    ("LC", 32),
    ("LI", 21),
    ("LT", 20),
    ("LU", 20),
    ("LV", 21),
    ("LY", 25),
    ("MC", 27),
    ("MD", 24),
    ("ME", 22),
    ("MK", 19),
    ("MN", 20),
    ("MR", 27),
    ("MT", 31),
    ("MU", 30),
    ("NI", 28),
    ("NL", 18),
    ("NO", 15),
    ("OM", 23),
    ("PK", 24),
    ("PL", 28),
    ("PS", 29),
    ("PT", 25),
    ("QA", 29),
    ("RO", 24),
    ("RS", 22),
    ("RU", 33),
    ("SA", 24),
    ("SC", 31),
    ("SD", 18),
    ("SE", 24),
    ("SI", 19),
    ("SK", 24),
    ("SM", 27),
    ("SO", 23),
    ("ST", 25),
    ("SV", 28),
    ("TL", 23),
    ("TN", 24),
    ("TR", 26),
    ("UA", 29),
    ("VA", 22),
    ("VG", 24),
    ("XK", 20),
    ("YE", 30),
];

/// IBAN check: known country, registry length, alphanumeric body and mod-97 == 1.
pub fn iban_valid(value: &str) -> bool {
    let compact: Vec<char> = value.chars().filter(|c| *c != ' ').collect();
    if compact.len() < 15 || !compact.iter().all(|c| c.is_ascii_alphanumeric()) {
        return false;
    }
    let country: String = compact[..2].iter().collect::<String>().to_ascii_uppercase();
    let Some(&(_, length)) = IBAN_LENGTHS.iter().find(|(cc, _)| *cc == country) else {
        return false;
    };
    if compact.len() != length || !compact[2].is_ascii_digit() || !compact[3].is_ascii_digit() {
        return false;
    }
    let mut remainder: u64 = 0;
    for c in compact[4..].iter().chain(compact[..4].iter()) {
        let c = c.to_ascii_uppercase();
        let v = if c.is_ascii_digit() {
            c as u64 - '0' as u64
        } else {
            c as u64 - 'A' as u64 + 10
        };
        remainder = if v >= 10 {
            (remainder * 100 + v) % 97
        } else {
            (remainder * 10 + v) % 97
        };
    }
    remainder == 1
}

/// RFC 5321 length limits and dot rules: local part 1..=64 characters, address at most 254,
/// no leading, trailing or doubled dots in either part, and an alphabetic top-level domain.
pub fn email_valid(value: &str) -> bool {
    if value.len() > 254 || value.matches('@').count() != 1 {
        return false;
    }
    let (local, domain) = value.split_once('@').unwrap_or(("", ""));
    if local.is_empty() || local.chars().count() > 64 || domain.is_empty() {
        return false;
    }
    for part in [local, domain] {
        if part.starts_with('.') || part.ends_with('.') || part.contains("..") {
            return false;
        }
    }
    let labels: Vec<&str> = domain.split('.').collect();
    labels.len() >= 2
        && labels.iter().all(|l| !l.is_empty())
        && labels
            .last()
            .is_some_and(|t| t.chars().all(|c| c.is_ascii_alphabetic()))
}

/// Dotted-quad IPv4: 1.0 for a public address, 0.6 for private, loopback, link-local, shared,
/// documentation and multicast ranges, `None` for invalid, unspecified or broadcast.
pub fn ipv4_factor(value: &str) -> Option<f64> {
    let mut octets = [0u32; 4];
    let mut n = 0;
    for part in value.split('.') {
        if n == 4 || part.is_empty() || part.len() > 3 || !part.bytes().all(|b| b.is_ascii_digit())
        {
            return None;
        }
        if part.len() > 1 && part.starts_with('0') {
            return None;
        }
        let v: u32 = part.parse().ok()?;
        if v > 255 {
            return None;
        }
        octets[n] = v;
        n += 1;
    }
    if n != 4 {
        return None;
    }
    let [a, b, c, _] = octets;
    if octets == [0, 0, 0, 0] || octets == [255, 255, 255, 255] {
        return None;
    }
    let reserved = a == 10
        || a == 127
        || (a == 172 && (16..=31).contains(&b))
        || (a == 192 && b == 168)
        || (a == 169 && b == 254)
        || (a == 100 && (64..=127).contains(&b))
        || (a == 192 && b == 0 && c == 2)
        || (a == 198 && b == 51 && c == 100)
        || (a == 203 && b == 0 && c == 113)
        || a >= 224
        || a == 0;
    Some(if reserved { 0.6 } else { 1.0 })
}

/// IPv6 with at least two colons: 1.0 for global, 0.6 for loopback, unique-local and
/// link-local, `None` if it does not parse or is unspecified.
pub fn ipv6_factor(value: &str) -> Option<f64> {
    if value.matches(':').count() < 2 {
        return None;
    }
    let addr: Ipv6Addr = value.parse().ok()?;
    if addr.is_unspecified() {
        return None;
    }
    let first = addr.segments()[0];
    let local = addr.is_loopback() || (first & 0xfe00) == 0xfc00 || (first & 0xffc0) == 0xfe80;
    Some(if local { 0.6 } else { 1.0 })
}

fn b64url_decode(part: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(part.len() * 3 / 4);
    let mut buf: u32 = 0;
    let mut bits = 0;
    for b in part.bytes() {
        let v = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'-' => 62,
            b'_' => 63,
            _ => return None,
        } as u32;
        buf = (buf << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }
    if bits >= 6 {
        return None;
    }
    Some(out)
}

/// JWT structure: three dot-separated base64url parts whose first two decode to JSON objects,
/// and the header names an `alg`.
pub fn jwt_valid(value: &str) -> bool {
    let parts: Vec<&str> = value.split('.').collect();
    if parts.len() != 3 || parts[0].is_empty() || parts[1].is_empty() {
        return false;
    }
    let (Some(header), Some(payload)) = (b64url_decode(parts[0]), b64url_decode(parts[1])) else {
        return false;
    };
    if !parts[2].is_empty() && b64url_decode(parts[2]).is_none() {
        return false;
    }
    let is_object = |bytes: &[u8]| {
        let s = String::from_utf8_lossy(bytes);
        let t = s.trim();
        t.starts_with('{') && t.ends_with('}')
    };
    is_object(&header)
        && is_object(&payload)
        && String::from_utf8_lossy(&header).contains("\"alg\"")
}

/// CRC-32 (IEEE 802.3, reflected, polynomial 0xEDB88320).
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

const BASE62: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// Base62 encoding left-padded with '0' to `width` characters.
pub fn base62(mut n: u64, width: usize) -> String {
    let mut out = Vec::new();
    while n > 0 {
        out.push(BASE62[(n % 62) as usize]);
        n /= 62;
    }
    while out.len() < width {
        out.push(b'0');
    }
    out.reverse();
    String::from_utf8(out).expect("ascii")
}

/// GitHub classic token checksum: `gh?_` + 30 random base62 characters + the base62 CRC-32 of
/// those 30 characters, padded to 6.
pub fn github_checksum_valid(value: &str) -> bool {
    let Some(body) = value.get(4..) else {
        return false;
    };
    if body.len() != 36 || !body.is_ascii() {
        return false;
    }
    let (random, check) = body.split_at(30);
    base62(crc32(random.as_bytes()) as u64, 6) == check
}

/// US SSN rules: area not 000, 666 or 900-999; group not 00; serial not 0000.
pub fn ssn_valid(value: &str) -> bool {
    let digits: String = value.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.len() != 9 {
        return false;
    }
    let area = &digits[..3];
    area != "000"
        && area != "666"
        && !area.starts_with('9')
        && &digits[3..5] != "00"
        && &digits[5..] != "0000"
}

/// Phone numbers have 7 to 15 digits (E.164 allows at most 15).
pub fn phone_factor(value: &str) -> Option<f64> {
    let n = value.chars().filter(|c| c.is_ascii_digit()).count();
    (7..=15).contains(&n).then_some(1.0)
}

/// Shannon entropy in bits per character, summed in code point order.
pub fn shannon_entropy(value: &str) -> f64 {
    let mut counts: BTreeMap<char, usize> = BTreeMap::new();
    let mut n = 0usize;
    for c in value.chars() {
        *counts.entry(c).or_insert(0) += 1;
        n += 1;
    }
    if n == 0 {
        return 0.0;
    }
    let mut h = 0.0f64;
    for &count in counts.values() {
        let p = count as f64 / n as f64;
        h -= p * p.log2();
    }
    h
}

const PLACEHOLDERS: &[&str] = &[
    "null",
    "none",
    "nil",
    "true",
    "false",
    "undefined",
    "password",
    "passwd",
    "secret",
    "changeme",
    "example",
    "redacted",
    "placeholder",
    "your_password",
    "your_api_key",
];

fn identifier(part: &str) -> bool {
    let mut chars = part.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {
            chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        }
        _ => false,
    }
}

/// `config.password` or `process.env.TOKEN`: code, not a secret.
fn dotted_identifier(value: &str) -> bool {
    let parts: Vec<&str> = value.split('.').collect();
    parts.len() >= 2 && parts.iter().all(|p| identifier(p))
}

/// Values assigned to secret-like keys: reject placeholders and templating, then score by
/// entropy (1.0 at 3.5 bits per character or more, 0.7 at 2.5 or more, else reject).
pub fn secret_value_factor(value: &str) -> Option<f64> {
    let lower = value.to_lowercase();
    let first = value.chars().next()?;
    if matches!(first, '$' | '{' | '<' | '%')
        || value.chars().any(|c| "()[]{}".contains(c))
        || dotted_identifier(value)
        || lower.contains("xxxx")
        || lower.contains("****")
        || PLACEHOLDERS.contains(&lower.as_str())
    {
        return None;
    }
    let h = shannon_entropy(value);
    if h >= 3.5 {
        Some(1.0)
    } else if h >= 2.5 {
        Some(0.7)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn email_rules() {
        use super::email_valid;
        assert!(email_valid("jane.doe@example.org"));
        assert!(email_valid("a+b@mail.example.co.uk"));
        assert!(!email_valid("jane..doe@example.org"));
        assert!(!email_valid(".jane@example.org"));
        assert!(!email_valid("jane@example.c0m"));
        assert!(!email_valid("jane@localhost"));
        assert!(!email_valid(&format!("{}@example.org", "a".repeat(65))));
    }

    use super::*;

    #[test]
    fn luhn() {
        assert!(luhn_valid("4111 1111 1111 1111"));
        assert!(luhn_valid("5500-0000-0000-0004"));
        assert!(!luhn_valid("4111 1111 1111 1112"));
        assert!(!luhn_valid("0000000000000000"));
        assert!(!luhn_valid("4111x1111"));
    }

    #[test]
    fn iban() {
        assert!(iban_valid("GB82 WEST 1234 5698 7654 32"));
        assert!(iban_valid("DE89370400440532013000"));
        assert!(!iban_valid("GB82 WEST 1234 5698 7654 33"));
        assert!(!iban_valid("ZZ82WEST12345698765432"));
    }

    #[test]
    fn ips() {
        assert_eq!(ipv4_factor("8.8.8.8"), Some(1.0));
        assert_eq!(ipv4_factor("192.168.1.10"), Some(0.6));
        assert_eq!(ipv4_factor("256.1.1.1"), None);
        assert_eq!(ipv4_factor("01.2.3.4"), None);
        assert_eq!(ipv6_factor("2001:4860:4860::8888"), Some(1.0));
        assert_eq!(ipv6_factor("fe80::1"), Some(0.6));
        assert_eq!(ipv6_factor("::"), None);
        assert_eq!(ipv6_factor("12:34"), None);
    }

    #[test]
    fn jwt() {
        // {"alg":"HS256","typ":"JWT"} . {"sub":"1234567890"} . signature
        let token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.c2lnbmF0dXJl";
        assert!(jwt_valid(token));
        assert!(!jwt_valid("eyJhbGciOiJIUzI1NiJ9.bm90IGpzb24.c2ln"));
        assert!(!jwt_valid("a.b"));
    }

    #[test]
    fn github_checksum_round_trip() {
        let random = "abcdefghijklmnopqrstuvwxyz0123";
        let token = format!("ghp_{random}{}", base62(crc32(random.as_bytes()) as u64, 6));
        assert!(github_checksum_valid(&token));
        assert!(!github_checksum_valid(&token.replace('a', "b")));
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn ssn_and_entropy() {
        assert!(ssn_valid("123-45-6789"));
        assert!(!ssn_valid("666-45-6789"));
        assert!(!ssn_valid("123-00-6789"));
        assert_eq!(shannon_entropy(""), 0.0);
        assert!((shannon_entropy("abcd") - 2.0).abs() < 1e-12);
        assert_eq!(secret_value_factor("${DB_PASSWORD}"), None);
        assert_eq!(secret_value_factor("aaaaaaaa"), None);
        assert_eq!(secret_value_factor("q8Zr2LmP0xTv9KcW"), Some(1.0));
        assert_eq!(check("unknown", "x"), None);
        assert_eq!(secret_value_factor("os.environ["), None);
        assert_eq!(secret_value_factor("settings.db_password"), None);
        assert_eq!(secret_value_factor("v1.2.3x9Qz"), Some(0.7));
    }
}
