"""
Certificate Generation and Management Module for Free Antigravity MITM Proxy.

This module handles all SSL/TLS certificate operations:
- Root CA generation: Creates a self-signed CA certificate and private key
- Root CA loading: Loads existing Root CA from disk (or generates new)
- Leaf certificate generation: Creates per-domain certificates signed by Root CA
- CA trust management: Installs/removes Root CA from system trust store

The MITM proxy needs to present certificates to clients that match the
hostname they're trying to connect to. Since we can't pre-generate
certificates for every possible domain, we generate them on-the-fly
signed by our Root CA.

Platform support:
- macOS: Uses Keychain (security command)
- Linux: Supports Debian/Ubuntu, Fedora/RHEL, Arch Linux
"""

# === Standard library imports ===
import datetime  # For setting certificate validity dates
import platform  # For detecting OS (Darwin/Linux)
import subprocess  # For running system commands (security, update-ca-certificates)
from pathlib import Path  # For filesystem paths
from typing import Tuple, Dict, Any  # Type hints

# === Internal module imports ===
from logger import get_logger  # Structured logging

# === Third-party imports (cryptography) ===
from cryptography import x509  # X.509 certificate handling
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID  # Certificate fields
from cryptography.hazmat.primitives import hashes, serialization  # Crypto primitives
from cryptography.hazmat.primitives.asymmetric import rsa  # RSA key generation


# =============================================================================
# IN-MEMORY CACHE
# =============================================================================

# Cache for generated leaf certificates to improve performance.
# Generating RSA keys is expensive, so we cache certificates by domain.
# Key: domain name (str), Value: (cert_pem_bytes, key_pem_bytes)
_leaf_cache: Dict[str, Tuple[bytes, bytes]] = {}

# Module-level logger
log = get_logger("cert")


# =============================================================================
# CERTIFICATE DIRECTORY MANAGEMENT
# =============================================================================


def _get_cert_dir(cert_dir: str) -> Path:
    """
    Get the certificate directory path, creating it if it doesn't exist.

    Args:
        cert_dir: Certificate directory path (may contain ~)

    Returns:
        Path object for the certificate directory
    """
    path = Path(cert_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# ROOT CA GENERATION
# =============================================================================


def generate_root_ca(cert_dir: str) -> Tuple[rsa.RSAPrivateKey, x509.Certificate]:
    """
    Generate a new Root CA certificate and private key.

    The Root CA is the trust anchor for all intercepting certificates.
    It has:
    - 10-year validity (3650 days)
    - CA constraint (can sign other certificates)
    - Key usage for certificate signing and CRL signing

    Files created:
    - rootCA.key: Private key (PEM format)
    - rootCA.crt: Certificate (PEM format)

    Args:
        cert_dir: Directory to save the CA files

    Returns:
        Tuple of (private_key, certificate) as cryptography objects
    """
    # Generate RSA key pair (2048-bit, public exponent 65537)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    # Create the certificate subject and issuer (same for self-signed CA)
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Free Antigravity"),
            x509.NameAttribute(NameOID.COMMON_NAME, "Free Antigravity Root CA"),
        ]
    )

    # Get current time for certificate validity
    now = datetime.datetime.now(datetime.timezone.utc)

    # Build the certificate
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=3650))  # 10 years
        # CA constraint: this certificate can sign other certificates
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        # Key usage: certificate signing and CRL signing
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
        .sign(key, hashes.SHA256())  # Sign with CA's own private key
    )

    # Save to disk
    path = _get_cert_dir(cert_dir)
    key_path = path / "rootCA.key"
    cert_path = path / "rootCA.crt"

    # Write private key in PEM format (unencrypted - root key is stored securely by user)
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    # Write certificate in PEM format
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    log.info("Root CA generated: {cert_path}", cert_path=cert_path)
    return key, cert


def load_or_create_root_ca(cert_dir: str) -> Tuple[Any, x509.Certificate]:
    """
    Load existing Root CA from disk, or generate a new one if not found.

    This is the main entry point for getting the Root CA. It checks
    if rootCA.key and rootCA.crt exist in the cert directory, and loads
    them if so. Otherwise, it generates new ones.

    Args:
        cert_dir: Directory containing CA files

    Returns:
        Tuple of (private_key, certificate)
    """
    path = _get_cert_dir(cert_dir)
    key_path = path / "rootCA.key"
    cert_path = path / "rootCA.crt"

    # Check if both files exist
    if key_path.exists() and cert_path.exists():
        # Load existing key and certificate
        key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        log.info("Root CA loaded from {cert_path}", cert_path=cert_path)
        return key, cert

    # Generate new CA if not found
    return generate_root_ca(cert_dir)


# =============================================================================
# LEAF CERTIFICATE GENERATION
# =============================================================================


def generate_leaf_cert(
    domain: str,
    ca_key: rsa.RSAPrivateKey,
    ca_cert: x509.Certificate,
) -> Tuple[bytes, bytes]:
    """
    Generate a leaf (server) certificate for a specific domain, signed by the Root CA.

    This is called during TLS handshake when a client connects.
    The certificate is generated on-the-fly for the domain being accessed.

    Certificate properties:
    - Valid for 1 year
    - Includes Subject Alternative Name (SAN) for the domain
    - Extended Key Usage for server authentication
    - Signed by our Root CA (clients must trust Root CA to accept this)

    Results are cached in memory for performance.

    Args:
        domain: The domain name (e.g., "cloudcode-pa.googleapis.com")
        ca_key: Root CA private key (used for signing)
        ca_cert: Root CA certificate (used for issuer info)

    Returns:
        Tuple of (cert_pem_bytes, key_pem_bytes)
    """
    # Check cache first to avoid regenerating for same domain
    if domain in _leaf_cache:
        return _leaf_cache[domain]

    # Generate new RSA key pair for this domain
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    # Set subject (who the certificate is for)
    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, domain),
        ]
    )

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)  # Issued by our Root CA
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=365))  # 1 year
        # SAN: Allows multiple domain names (important for some services)
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(domain)]),
            critical=False,
        )
        # EKU: Server authentication
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())  # Sign with Root CA key
    )

    # Convert to PEM format (text encoding of binary cert)
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )

    # Cache for future requests
    _leaf_cache[domain] = (cert_pem, key_pem)
    return cert_pem, key_pem


# =============================================================================
# CA FILE MANAGEMENT
# =============================================================================


def get_ca_cert_path(cert_dir: str) -> Path:
    """Get the path to the Root CA certificate file."""
    return Path(cert_dir).expanduser() / "rootCA.crt"


def get_ca_key_path(cert_dir: str) -> Path:
    """Get the path to the Root CA private key file."""
    return Path(cert_dir).expanduser() / "rootCA.key"


def ca_exists(cert_dir: str) -> bool:
    """
    Check if Root CA certificate files exist.

    Args:
        cert_dir: Directory containing CA files

    Returns:
        True if both rootCA.key and rootCA.crt exist
    """
    cert_path = get_ca_cert_path(cert_dir)
    key_path = get_ca_key_path(cert_dir)
    return cert_path.exists() and key_path.exists()


# =============================================================================
# TRUST STORE MANAGEMENT
# =============================================================================


def is_trusted(cert_dir: str) -> bool:
    """
    Check if the Root CA is trusted by the operating system.

    Different platforms have different trust mechanisms:
    - macOS: System keychain with trust settings
    - Linux: Various directories and system certificate bundles

    Args:
        cert_dir: Directory containing CA files

    Returns:
        True if the CA is trusted, False otherwise
    """
    cert_path = get_ca_cert_path(cert_dir)
    if not cert_path.exists():
        return False

    system = platform.system()

    # === macOS ===
    if system == "Darwin":
        try:
            # Check if cert exists in System keychain
            result = subprocess.run(
                [
                    "security",
                    "find-certificate",
                    "-c",
                    "Free Antigravity Root CA",
                    "-p",
                ],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except Exception:
            return False

    # === Linux ===
    elif system == "Linux":
        # Check common CA certificate directories
        cert_name = cert_path.name
        dirs_to_check = [
            "/usr/local/share/ca-certificates",  # Debian/Ubuntu
            "/etc/ca-certificates/trust-source/anchors",  # Arch Linux
            "/etc/pki/ca-trust/source/anchors",  # Fedora/RHEL
            "/etc/ssl/certs",  # Generic
        ]

        for ca_dir in dirs_to_check:
            ca_path = Path(ca_dir)
            if ca_path.exists():
                # Check if cert file exists (with or without .crt extension)
                if (ca_path / cert_name).exists() or (
                    ca_path / f"{cert_name}.crt"
                ).exists():
                    return True

        # Check system certificate bundle
        try:
            result = subprocess.run(
                ["update-ca-certificates", "--help"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                # Try to find cert in system bundle
                system_cert_bundle = "/etc/ssl/certs/ca-certificates.crt"
                if Path(system_cert_bundle).exists():
                    cert_content = cert_path.read_bytes()
                    bundle_content = Path(system_cert_bundle).read_bytes()
                    if cert_content in bundle_content:
                        return True
        except Exception:
            pass

        return False

    return False


def trust_ca(cert_dir: str, force: bool = False) -> bool:
    """
    Install and trust the Root CA in the system trust store.

    This makes the system trust certificates signed by our Root CA,
    allowing the MITM proxy to intercept HTTPS traffic without
    certificate warnings.

    Args:
        cert_dir: Directory containing CA files
        force: If True, reinstall even if already trusted

    Returns:
        True if successful, False otherwise
    """
    if not ca_exists(cert_dir):
        log.error("Root CA does not exist. Run generate_root_ca first.")
        return False

    if is_trusted(cert_dir) and not force:
        log.info("Root CA is already trusted.")
        return True

    cert_path = get_ca_cert_path(cert_dir)
    system = platform.system()

    try:
        # === macOS ===
        if system == "Darwin":
            log.banner("Installing Root CA to macOS System keychain...")
            # Add to System keychain with trust settings
            subprocess.run(
                [
                    "security",
                    "add-trusted-cert",
                    "-d",  # Domain trust (not SSL policy)
                    "-r",
                    "trustRoot",  # Trust as root CA
                    "-k",
                    "/Library/Keychains/System.keychain",  # System keychain
                    str(cert_path),
                ],
                check=True,
            )
            log.success("Root CA trusted on macOS.")

        # === Linux ===
        elif system == "Linux":
            log.banner("Installing Root CA to Linux trust store...")

            # Detect distribution and install accordingly

            # Debian/Ubuntu
            if Path("/etc/debian_version").exists():
                dest_dir = "/usr/local/share/ca-certificates"
                # Copy cert (must have .crt extension)
                subprocess.run(["cp", str(cert_path), dest_dir], check=True)
                # Update system CA certificates
                subprocess.run(["update-ca-certificates"], check=True)
                log.success("Root CA installed to {dest_dir}/", dest_dir=dest_dir)

            # Fedora/RHEL
            elif (
                Path("/etc/fedora-release").exists()
                or Path("/etc/redhat-release").exists()
            ):
                dest_dir = "/etc/pki/ca-trust/source/anchors"
                subprocess.run(["cp", str(cert_path), dest_dir], check=True)
                subprocess.run(["update-ca-trust"], check=True)
                log.success("Root CA installed to {dest_dir}/", dest_dir=dest_dir)

            # Arch Linux
            elif Path("/etc/arch-release").exists():
                dest_dir = "/etc/ca-certificates/trust-source/anchors"
                subprocess.run(["cp", str(cert_path), dest_dir], check=True)
                subprocess.run(["trust", "extract-compat"], check=True)
                log.success("Root CA installed to {dest_dir}/", dest_dir=dest_dir)

            else:
                # Generic Linux fallback
                log.warning("Unknown Linux distribution. Attempting generic install...")
                dest_dir = "/usr/local/share/ca-certificates"
                if Path(dest_dir).exists():
                    subprocess.run(["cp", str(cert_path), dest_dir], check=True)
                    subprocess.run(["update-ca-certificates"], check=True)
                    log.success("Root CA installed to {dest_dir}/", dest_dir=dest_dir)
                else:
                    log.error("Cannot find CA certificate directory.")
                    return False

        else:
            log.error("Unsupported platform: {system}", system=system)
            return False

        log.info("Please restart your browser/application to pick up the new CA.")
        return True

    except subprocess.CalledProcessError as e:
        log.error("Command failed with return code {rc}", rc=e.returncode)
        return False
    except PermissionError:
        log.error("Root privileges required. Run with sudo.")
        return False
    except Exception as e:
        log.error("{e}", e=e)
        return False


def untrust_ca(cert_dir: str) -> bool:
    """
    Remove the Root CA from the system trust store.

    The certificate files are NOT deleted — only the trust
    relationship is removed.

    Args:
        cert_dir: Directory containing CA files

    Returns:
        True if successful, False otherwise
    """
    cert_path = get_ca_cert_path(cert_dir)
    system = platform.system()

    try:
        # === macOS ===
        if system == "Darwin":
            log.banner("Removing Root CA from macOS keychain...")
            # Find and remove the certificate
            result = subprocess.run(
                [
                    "security",
                    "find-certificate",
                    "-c",
                    "Free Antigravity Root CA",
                    "-p",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                # Remove using the cert directly
                subprocess.run(
                    ["security", "remove-trusted-cert", "-d", str(cert_path)],
                    check=True,
                )
                log.success("Root CA removed from macOS keychain.")
            else:
                log.info("Root CA not found in keychain.")

        # === Linux ===
        elif system == "Linux":
            log.banner("Removing Root CA from Linux trust store...")

            # Debian/Ubuntu
            if Path("/etc/debian_version").exists():
                cert_name = cert_path.name
                src = f"/usr/local/share/ca-certificates/{cert_name}"
                if Path(src).exists():
                    Path(src).unlink()
                    subprocess.run(["update-ca-certificates"], check=True)
                    log.success("Removed {src}", src=src)

            # Fedora/RHEL
            elif (
                Path("/etc/fedora-release").exists()
                or Path("/etc/redhat-release").exists()
            ):
                cert_name = cert_path.name
                src = f"/etc/pki/ca-trust/source/anchors/{cert_name}"
                if Path(src).exists():
                    Path(src).unlink()
                    subprocess.run(["update-ca-trust"], check=True)
                    log.success("Removed {src}", src=src)

            # Arch Linux
            elif Path("/etc/arch-release").exists():
                cert_name = cert_path.name
                src = f"/etc/ca-certificates/trust-source/anchors/{cert_name}"
                if Path(src).exists():
                    Path(src).unlink()
                    subprocess.run(["trust", "extract-compat"], check=True)
                    log.success("Removed {src}", src=src)

            else:
                log.warning("Unknown Linux distribution. Please remove manually.")

        else:
            log.error("Unsupported platform: {system}", system=system)
            return False

        log.info("Please restart your browser/application.")
        return True

    except subprocess.CalledProcessError as e:
        log.error("Command failed with return code {rc}", rc=e.returncode)
        return False
    except PermissionError:
        log.error("Root privileges required. Run with sudo.")
        return False
    except Exception as e:
        log.error("{e}", e=e)
        return False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def install_ca(cert_dir: str) -> bool:
    """
    Generate (if needed) and install the Root CA.

    This is a convenience function that combines generation and trust installation.

    Args:
        cert_dir: Directory to store/load the CA files

    Returns:
        True if successful, False otherwise
    """
    log.banner("Setting up Root CA...")

    # Ensure CA exists (generate if not)
    load_or_create_root_ca(cert_dir)

    # Trust it
    return trust_ca(cert_dir)


def uninstall_ca(cert_dir: str, delete_files: bool = False) -> bool:
    """
    Uninstall the Root CA from the system.

    Args:
        cert_dir: Directory containing CA files
        delete_files: If True, also delete the CA certificate files

    Returns:
        True if successful, False otherwise
    """
    log.banner("Uninstalling Root CA...")

    # Remove from trust store
    untrust_ca(cert_dir)

    # Optionally delete files
    if delete_files:
        cert_path = get_ca_cert_path(cert_dir)
        key_path = get_ca_key_path(cert_dir)

        if cert_path.exists():
            cert_path.unlink()
            log.success("Deleted {cert_path}", cert_path=cert_path)

        if key_path.exists():
            key_path.unlink()
            log.success("Deleted {key_path}", key_path=key_path)

        log.success("Root CA files deleted.")

    return True
