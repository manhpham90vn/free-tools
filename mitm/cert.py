"""
Certificate generation module for Antigravity MITM proxy.
Generates Root CA and leaf certificates for TLS interception.
"""

import datetime
from pathlib import Path
from typing import Tuple, Dict, Any

from cryptography import x509
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa


# In-memory cache for leaf certs
_leaf_cache: Dict[str, Tuple[bytes, bytes]] = {}


def _get_cert_dir(cert_dir: str) -> Path:
    """Get and ensure cert directory exists."""
    path = Path(cert_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_root_ca(cert_dir: str) -> Tuple[rsa.RSAPrivateKey, x509.Certificate]:
    """
    Generate a new Root CA certificate and key.

    Args:
        cert_dir: Directory to store the CA files

    Returns:
        Tuple of (private_key, certificate)
    """
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Free Antigravity"),
            x509.NameAttribute(NameOID.COMMON_NAME, "Free Antigravity Root CA"),
        ]
    )

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_cert_sign=True,
                crl_sign=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(key, hashes.SHA256())
    )

    # Save to disk
    path = _get_cert_dir(cert_dir)
    key_path = path / "rootCA.key"
    cert_path = path / "rootCA.crt"

    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    print(f"Root CA generated: {cert_path}")
    return key, cert


def load_or_create_root_ca(cert_dir: str) -> Tuple[Any, x509.Certificate]:
    """
    Load existing Root CA from disk, or create a new one.

    Args:
        cert_dir: Directory containing CA files

    Returns:
        Tuple of (private_key, certificate)
    """
    path = _get_cert_dir(cert_dir)
    key_path = path / "rootCA.key"
    cert_path = path / "rootCA.crt"

    if key_path.exists() and cert_path.exists():
        key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        print(f"Root CA loaded from {cert_path}")
        return key, cert

    return generate_root_ca(cert_dir)


def generate_leaf_cert(
    domain: str,
    ca_key: rsa.RSAPrivateKey,
    ca_cert: x509.Certificate,
) -> Tuple[bytes, bytes]:
    """
    Generate a leaf certificate for a specific domain, signed by the Root CA.
    Results are cached in memory.

    Args:
        domain: The domain name for the certificate
        ca_key: Root CA private key
        ca_cert: Root CA certificate

    Returns:
        Tuple of (cert_pem_bytes, key_pem_bytes)
    """
    if domain in _leaf_cache:
        return _leaf_cache[domain]

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, domain),
        ]
    )

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(domain)]),
            critical=False,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )

    _leaf_cache[domain] = (cert_pem, key_pem)
    return cert_pem, key_pem
