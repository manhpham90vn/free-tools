"""
Certificate generation module for Antigravity MITM proxy.
Generates Root CA and leaf certificates for TLS interception.
"""

import datetime
import platform
import subprocess
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


# =============================================================================
# CA Management Functions (install, trust, uninstall)
# =============================================================================


def get_ca_cert_path(cert_dir: str) -> Path:
    """Get the path to the Root CA certificate."""
    return Path(cert_dir).expanduser() / "rootCA.crt"


def get_ca_key_path(cert_dir: str) -> Path:
    """Get the path to the Root CA private key."""
    return Path(cert_dir).expanduser() / "rootCA.key"


def ca_exists(cert_dir: str) -> bool:
    """Check if Root CA exists."""
    cert_path = get_ca_cert_path(cert_dir)
    key_path = get_ca_key_path(cert_dir)
    return cert_path.exists() and key_path.exists()


def is_trusted(cert_dir: str) -> bool:
    """
    Check if the Root CA is trusted by the system.

    Returns:
        True if the CA is trusted, False otherwise.
    """
    cert_path = get_ca_cert_path(cert_dir)
    if not cert_path.exists():
        return False

    system = platform.system()

    if system == "Darwin":
        # Check if cert exists in System keychain with trust settings
        try:
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

    elif system == "Linux":
        # Check common CA certificate directories
        cert_name = cert_path.name
        dirs_to_check = [
            "/usr/local/share/ca-certificates",
            "/etc/ca-certificates/trust-source/anchors",
            "/etc/pki/ca-trust/source/anchors",
            "/etc/ssl/certs",
        ]

        for ca_dir in dirs_to_check:
            ca_path = Path(ca_dir)
            if ca_path.exists():
                # Check if cert file exists (or .crt extension variant)
                if (ca_path / cert_name).exists() or (
                    ca_path / f"{cert_name}.crt"
                ).exists():
                    return True

        # Check for certificate in system bundle
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
    Requires root privileges.

    Args:
        cert_dir: Directory containing CA files
        force: If True, reinstall even if already trusted

    Returns:
        True if successful, False otherwise.
    """
    if not ca_exists(cert_dir):
        print("Error: Root CA does not exist. Run generate_root_ca first.")
        return False

    if is_trusted(cert_dir) and not force:
        print("[*] Root CA is already trusted.")
        return True

    cert_path = get_ca_cert_path(cert_dir)
    system = platform.system()

    try:
        if system == "Darwin":
            # macOS: Add to System keychain with trust
            print("[*] Installing Root CA to macOS System keychain...")
            subprocess.run(
                [
                    "security",
                    "add-trusted-cert",
                    "-d",
                    "-r",
                    "trustRoot",
                    "-k",
                    "/Library/Keychains/System.keychain",
                    str(cert_path),
                ],
                check=True,
            )
            print("[+] Root CA trusted on macOS.")

        elif system.startswith("linux"):
            # Linux: Copy to appropriate directory based on distribution
            print("[*] Installing Root CA to Linux trust store...")

            # Detect distribution and install accordingly
            if Path("/etc/debian_version").exists():
                # Debian/Ubuntu
                dest_dir = "/usr/local/share/ca-certificates"
                subprocess.run(["cp", str(cert_path), dest_dir], check=True)
                subprocess.run(["update-ca-certificates"], check=True)
                print(f"[+] Root CA installed to {dest_dir}/")

            elif (
                Path("/etc/fedora-release").exists()
                or Path("/etc/redhat-release").exists()
            ):
                # Fedora/RHEL
                dest_dir = "/etc/pki/ca-trust/source/anchors"
                subprocess.run(["cp", str(cert_path), dest_dir], check=True)
                subprocess.run(["update-ca-trust"], check=True)
                print(f"[+] Root CA installed to {dest_dir}/")

            elif Path("/etc/arch-release").exists():
                # Arch Linux
                dest_dir = "/etc/ca-certificates/trust-source/anchors"
                subprocess.run(["cp", str(cert_path), dest_dir], check=True)
                subprocess.run(["trust", "extract-compat"], check=True)
                print(f"[+] Root CA installed to {dest_dir}/")

            else:
                # Generic Linux
                print("[!] Unknown Linux distribution. Attempting generic install...")
                dest_dir = "/usr/local/share/ca-certificates"
                if Path(dest_dir).exists():
                    subprocess.run(["cp", str(cert_path), dest_dir], check=True)
                    subprocess.run(["update-ca-certificates"], check=True)
                    print(f"[+] Root CA installed to {dest_dir}/")
                else:
                    print("Error: Cannot find CA certificate directory.")
                    return False

        else:
            print(f"Error: Unsupported platform: {system}")
            return False

        print("[*] Please restart your browser/application to pick up the new CA.")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with return code {e.returncode}")
        return False
    except PermissionError:
        print("Error: Root privileges required. Run with sudo.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def untrust_ca(cert_dir: str) -> bool:
    """
    Remove the Root CA from the system trust store.
    Requires root privileges.

    Args:
        cert_dir: Directory containing CA files

    Returns:
        True if successful, False otherwise.
    """
    cert_path = get_ca_cert_path(cert_dir)
    system = platform.system()

    try:
        if system == "Darwin":
            # macOS: Remove from System keychain
            print("[*] Removing Root CA from macOS keychain...")

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
                print("[+] Root CA removed from macOS keychain.")
            else:
                print("[*] Root CA not found in keychain.")

        elif system.startswith("linux"):
            # Linux: Remove from CA certificate directories
            print("[*] Removing Root CA from Linux trust store...")

            if Path("/etc/debian_version").exists():
                # Debian/Ubuntu
                cert_name = cert_path.name
                src = f"/usr/local/share/ca-certificates/{cert_name}"
                if Path(src).exists():
                    Path(src).unlink()
                    subprocess.run(["update-ca-certificates"], check=True)
                    print(f"[+] Removed {src}")

            elif (
                Path("/etc/fedora-release").exists()
                or Path("/etc/redhat-release").exists()
            ):
                # Fedora/RHEL
                cert_name = cert_path.name
                src = f"/etc/pki/ca-trust/source/anchors/{cert_name}"
                if Path(src).exists():
                    Path(src).unlink()
                    subprocess.run(["update-ca-trust"], check=True)
                    print(f"[+] Removed {src}")

            elif Path("/etc/arch-release").exists():
                # Arch Linux
                cert_name = cert_path.name
                src = f"/etc/ca-certificates/trust-source/anchors/{cert_name}"
                if Path(src).exists():
                    Path(src).unlink()
                    subprocess.run(["trust", "extract-compat"], check=True)
                    print(f"[+] Removed {src}")

            else:
                print("[!] Unknown Linux distribution. Please remove manually.")

        else:
            print(f"Error: Unsupported platform: {system}")
            return False

        print("[*] Please restart your browser/application.")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with return code {e.returncode}")
        return False
    except PermissionError:
        print("Error: Root privileges required. Run with sudo.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def install_ca(cert_dir: str) -> bool:
    """
    Generate (if needed) and install the Root CA.
    Wrapper around generate, trust functions.

    Args:
        cert_dir: Directory to store/load the CA files

    Returns:
        True if successful, False otherwise.
    """
    print("[*] Setting up Root CA...")

    # Ensure CA exists (generate if not)
    load_or_create_root_ca(cert_dir)

    # Trust it
    return trust_ca(cert_dir)


def uninstall_ca(cert_dir: str, delete_files: bool = False) -> bool:
    """
    Uninstall the Root CA from the system trust store.

    Args:
        cert_dir: Directory containing CA files
        delete_files: If True, also delete the CA certificate files

    Returns:
        True if successful, False otherwise.
    """
    print("[*] Uninstalling Root CA...")

    # Remove from trust store
    untrust_ca(cert_dir)

    # Optionally delete files
    if delete_files:
        cert_path = get_ca_cert_path(cert_dir)
        key_path = get_ca_key_path(cert_dir)

        if cert_path.exists():
            cert_path.unlink()
            print(f"[*] Deleted {cert_path}")

        if key_path.exists():
            key_path.unlink()
            print(f"[*] Deleted {key_path}")

        print("[*] Root CA files deleted.")

    return True
