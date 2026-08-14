#!/usr/bin/env bash
#
# Build the .deb and .rpm.
#
# Deliberately a CPU-only build. ONNX Runtime's CUDA provider is linked against
# a specific CUDA major version and ships as separate shared objects that are
# not ours to redistribute, so a packaged Murmur transcribes on the CPU — which
# is fast enough — and anyone wanting the GPU builds with `--features cuda` and
# supplies the runtime themselves. See the GPU section of the README.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$here")"
out="${1:-$root/target/packages}"

cd "$root"
mkdir -p "$out"

echo "==> building release binaries (cpu)"
cargo build --release -p murmur-cli -p murmur-hud

# Everything shipped is generated from the source it is generated from, so a
# stale icon in the tree cannot reach a package unnoticed.
echo "==> checking packaged assets are current"
cargo test -p murmur-hud --quiet icon::tests::the_installed_icon_matches_what_the_code_draws

echo "==> deb"
( cd crates/murmur-cli && cargo deb --no-build --output "$out" )

echo "==> rpm"
version="$(grep -m1 '^version' "$root/Cargo.toml" | cut -d'"' -f2)"
( cd crates/murmur-cli && cargo generate-rpm --output "$out/murmur-${version}-1.x86_64.rpm" )

echo
echo "built into $out:"
ls -la "$out" | tail -n +2 | awk '{printf "  %-44s %s bytes\n", $NF, $5}'
echo
echo "install with one of:"
echo "  sudo apt install $out/murmur_${version}-1_amd64.deb"
echo "  sudo dnf install $out/murmur-${version}-1.x86_64.rpm"
