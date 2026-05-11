# bot-cpp

> C++ runtime of the Hanzo Brain. Header-only C++17. Embeds in any C/C++ host.

Same algorithm surface as the TypeScript canonical
([`hanzoai/brain`](https://github.com/hanzoai/brain), `@hanzo/bot-memory`),
Python ([`hanzoai/python-sdk`](https://github.com/hanzoai/python-sdk),
`hanzo-memory`), Go ([`hanzoai/bot-go`](https://github.com/hanzoai/bot-go)),
and Rust ([`hanzoai/mcp`](https://github.com/hanzoai/mcp),
`hanzo_mcp::brain::algorithms`).

A `~/.hanzo/brain/brain.db` written by any of those runtimes is read by this
header-only library without translation.

## Use

```cpp
#include <hanzo/brain/algorithms.hpp>

using namespace hanzo::brain;

int main() {
    // Hybrid search → RRF fusion
    auto results = rrf_fuse({fts_hits, dense_hits}, /*limit=*/20);

    // MMR rerank for diversity
    auto diverse = mmr_rerank(std::move(mmr_inputs), /*lambda=*/0.5, /*limit=*/10);

    // Adaptive RRF k
    auto k = select_rrf_k(characterize(query));

    // Wallet-style content address (compatible with hanzo:* and mm:* prefixes)
    auto addr = encode_address(public_key, "hanzo");
}
```

## Build

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## What ships

`include/hanzo/brain/algorithms.hpp` — header-only C++17 port of the pure-CPU
brain primitives. One external dep: BLAKE3 (pulled at configure time from
[`luxcpp/blake3-reference`](https://github.com/luxcpp/blake3-reference) — the
canonical Lux C++ crypto source). No vendored copies.

- **Retrieval**: `rrf_fuse`, `rsf_fuse`, `select_rrf_k`, `select_weights`,
  `mmr_rerank`, `cosine`, `dedup_hits`
- **Text / FTS**: `detect_script`, `has_cjk`, `has_emoji`, `cjk_bigrams`,
  `emoji_trigrams`, `parse_websearch`, `to_fts5_match`
- **Embed**: `EmbeddingModel` registry, `prefix_for` (E5 asymmetric prefixes),
  `mrl_truncate` (Matryoshka), `coarse_dim`, `l2_normalize`
- **Temporal**: `v7_floor` / `v7_ceiling` (UUIDv7 bounds)
- **Captions**: `render_vtt`, `render_srt`, `render_rttm`
- **Tokenizer**: `estimate_tokens`, `truncate_to_tokens`
- **Eval**: `reciprocal_rank`, `mean_reciprocal_rank`, `recall_at_k`,
  `precision_at_k`, `ndcg_at_k`
- **Spatial**: `haversine_km`, `bbox_around`, `in_box`
- **HTTP Range**: `parse_range`, `content_range`
- **Crypto**: `encode_address`, `decode_address` (wallet-style content-addressable
  ids — real BLAKE3 via `luxcpp/blake3-reference`, byte-equivalent with TS / Python / Go / Rust)
- **Graph**: `normalize_edges`, `snn_score`, `pfnet_infinity`, `louvain`
- **Inference**: `parse_slug`, `format_slug`, `RuntimeConfig` (db_override → env
  → default), `classify_link_rule`
- **Resilience**: `CircuitBreaker`, `CircuitOpenError`

## Wires into the Hanzo stack

The brain rides on top of Hanzo's distributed-SQL stack and threshold crypto:

- **Transport**: ZAP ([`zap-proto`](https://github.com/zap-proto)).
- **Consensus**: [`hanzo-consensus`](https://github.com/hanzoai/python-sdk/tree/main/pkg/hanzo-consensus)
  — metastable agreement; also usable for storage quorum.
- **Native store**: `zapdb` at `zap-proto/db` (Go + C++ ports). The brain default
  is SQLite + FTS5 for solo use; scale-out flips to zapdb behind the same
  `BrainStore` contract.
- **Threshold crypto / sealing**: [`hanzoai/mpc`](https://github.com/hanzoai/mpc)
  for MPC-backed signing of brain envelopes and recipient keys; the
  `MMPKE01` envelope shape in this header maps 1:1 onto our threshold-signed
  recipient blocks.
- **Secrets**: [`hanzoai/kms`](https://github.com/hanzoai/kms) for at-rest
  secret material the brain depends on (replicate WAL keys, embedding API
  tokens, etc.).

C++ consumers that want full threshold signing link `hanzoai/mpc` (gRPC/HTTP
client) alongside this header; the algorithm primitives stay pure-CPU.

## Tests

```bash
cmake -S . -B build && cmake --build build && build/test_algorithms
```

Single test runner. Same cases the other runtimes use. Run them, compare the
output bytes.

## License

MIT.
