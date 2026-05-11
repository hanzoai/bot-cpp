// hanzo/brain/algorithms.hpp
//
// Header-only C++17 port of the Hanzo Brain pure-CPU algorithm set.
// Mirrors @hanzo/bot-memory (TS canonical), hanzo_memory.algorithms (Python),
// pkg/brain (Go), and hanzo_mcp::brain::algorithms (Rust).
//
// Outputs are byte-equivalent across runtimes for deterministic algorithms
// (slugify, base58, fusion ranking, captions, etc.) modulo floating-point
// noise in cosine-based code. Tested via tests/test_algorithms.cpp.
//
// The brain itself rides on Hanzo's distributed-SQL stack:
//   ZAP transport + hanzo-consensus + zapdb (zap-proto/db)
// — and the wallet-style identity + envelope crypto is meant to plug
// into the hanzoai/mpc threshold service (or hanzoai/kms) for production
// signing / sealing. The algorithms here are the pure-CPU primitives;
// network and threshold layers are not in this header.

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace hanzo::brain {

// ── Common types ─────────────────────────────────────────────────────

struct SearchHit {
    std::string slug;
    double score = 0.0;
    std::string excerpt;
    std::string source;
};

// ── Fusion ───────────────────────────────────────────────────────────

constexpr double RRF_K_DEFAULT = 20.0;

inline std::vector<SearchHit> rrf_fuse(const std::vector<std::vector<SearchHit>>& lists,
                                      std::size_t limit,
                                      double k = RRF_K_DEFAULT) {
    std::unordered_map<std::string, double> scores;
    std::unordered_map<std::string, SearchHit> meta;
    const double num = static_cast<double>(lists.size());
    for (const auto& lst : lists) {
        for (std::size_t rank = 0; rank < lst.size(); ++rank) {
            const auto& h = lst[rank];
            scores[h.slug] += 1.0 / (k + static_cast<double>(rank) + 1.0);
            meta.try_emplace(h.slug, h);
        }
    }
    if (scores.empty()) return {};
    const double max = num / (k + 1.0);
    std::vector<SearchHit> out;
    out.reserve(scores.size());
    for (auto& [slug, s] : scores) {
        const double norm = max > 0.0 ? std::min(s / max, 1.0) : 0.0;
        SearchHit h = meta[slug];
        h.score = norm;
        h.source = "fused";
        out.push_back(std::move(h));
    }
    std::sort(out.begin(), out.end(),
              [](const SearchHit& a, const SearchHit& b) { return a.score > b.score; });
    if (out.size() > limit) out.resize(limit);
    return out;
}

inline std::vector<SearchHit> rsf_fuse(const std::vector<std::vector<SearchHit>>& lists,
                                      std::size_t limit,
                                      const std::vector<double>& weights = {}) {
    const auto n = lists.size();
    std::vector<double> w = weights;
    if (w.empty()) w = std::vector<double>(n, n == 0 ? 0.0 : 1.0 / static_cast<double>(n));
    if (w.size() != n) w = std::vector<double>(n, n == 0 ? 0.0 : 1.0 / static_cast<double>(n));
    std::unordered_map<std::string, double> scores;
    std::unordered_map<std::string, SearchHit> meta;
    for (std::size_t i = 0; i < n; ++i) {
        const auto& lst = lists[i];
        if (lst.empty()) continue;
        double lo = std::numeric_limits<double>::infinity();
        double hi = -std::numeric_limits<double>::infinity();
        for (const auto& h : lst) { if (h.score < lo) lo = h.score; if (h.score > hi) hi = h.score; }
        const double span = hi - lo;
        for (const auto& h : lst) {
            const double norm = span > 0.0 ? (h.score - lo) / span : 1.0;
            scores[h.slug] += w[i] * norm;
            meta.try_emplace(h.slug, h);
        }
    }
    std::vector<SearchHit> out;
    out.reserve(scores.size());
    for (auto& [slug, s] : scores) {
        SearchHit h = meta[slug];
        h.score = s;
        h.source = "fused";
        out.push_back(std::move(h));
    }
    std::sort(out.begin(), out.end(),
              [](const SearchHit& a, const SearchHit& b) { return a.score > b.score; });
    if (out.size() > limit) out.resize(limit);
    return out;
}

struct QueryCharacteristics {
    std::size_t token_count = 0;
    bool is_phrase = false;
    bool is_boolean = false;
};

inline QueryCharacteristics characterize(const std::string& query) {
    QueryCharacteristics q;
    std::string t = query;
    // trim
    auto not_ws = [](char c) { return !std::isspace(static_cast<unsigned char>(c)); };
    t.erase(t.begin(), std::find_if(t.begin(), t.end(), not_ws));
    t.erase(std::find_if(t.rbegin(), t.rend(), not_ws).base(), t.end());

    q.is_phrase = (t.size() >= 2 && t.front() == '"' && t.back() == '"')
               || (t.size() >= 2 && t.front() == '\'' && t.back() == '\'');

    static const std::regex bool_re("\\b(AND|OR|NOT)\\b");
    static const std::regex neg_re("\\s-\\S");
    q.is_boolean = std::regex_search(t, bool_re) || std::regex_search(t, neg_re);

    std::istringstream iss(t);
    std::string tok;
    while (iss >> tok) ++q.token_count;
    return q;
}

inline int select_rrf_k(const QueryCharacteristics& q) {
    if (q.is_phrase) return 10;
    if (q.is_boolean) return 15;
    if (q.token_count <= 2) return 15;
    if (q.token_count >= 10) return 40;
    return 20;
}

struct FusionWeights { double fts; double semantic; };

inline FusionWeights select_weights(const QueryCharacteristics& q) {
    if (q.is_phrase)        return {0.8, 0.2};
    if (q.is_boolean)       return {0.7, 0.3};
    if (q.token_count <= 2) return {0.65, 0.35};
    if (q.token_count >= 10) return {0.3, 0.7};
    return {0.5, 0.5};
}

// ── Rerank (MMR) ─────────────────────────────────────────────────────

inline double cosine(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return 0.0;
    double dot = 0, na = 0, nb = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    const double d = std::sqrt(na) * std::sqrt(nb);
    return d > 0.0 ? dot / d : 0.0;
}

struct MmrInput {
    SearchHit hit;
    std::optional<std::vector<double>> embedding;
};

inline std::vector<MmrInput> mmr_rerank(std::vector<MmrInput> hits, double lambda, std::size_t limit) {
    if (limit == 0) limit = hits.size();
    std::vector<MmrInput> embedded;
    std::vector<MmrInput> orphans;
    embedded.reserve(hits.size());
    for (auto& h : hits) {
        if (h.embedding && !h.embedding->empty()) embedded.push_back(std::move(h));
        else orphans.push_back(std::move(h));
    }
    std::vector<MmrInput> selected;
    selected.reserve(limit);
    while (selected.size() < limit && !embedded.empty()) {
        std::size_t best_idx = static_cast<std::size_t>(-1);
        double best_score = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < embedded.size(); ++i) {
            const auto& c = embedded[i];
            double rel = c.hit.score;
            double max_sim = 0.0;
            for (const auto& s : selected) {
                const double sim = cosine(*c.embedding, *s.embedding);
                if (sim > max_sim) max_sim = sim;
            }
            const double mmr = lambda * rel - (1.0 - lambda) * max_sim;
            if (mmr > best_score) { best_score = mmr; best_idx = i; }
        }
        if (best_idx == static_cast<std::size_t>(-1)) break;
        selected.push_back(std::move(embedded[best_idx]));
        embedded.erase(embedded.begin() + static_cast<std::ptrdiff_t>(best_idx));
    }
    for (auto& o : orphans) {
        if (selected.size() >= limit) break;
        selected.push_back(std::move(o));
    }
    return selected;
}

// ── Dedup ────────────────────────────────────────────────────────────

inline std::string chain_of(const std::string& slug) {
    static const std::regex chunk_re("(#chunk-\\d+|::\\d+)$");
    return std::regex_replace(slug, chunk_re, "");
}

inline std::vector<SearchHit> dedup_hits(std::vector<SearchHit> hits, std::size_t per_chain = 1) {
    if (per_chain == 0) per_chain = 1;
    std::unordered_map<std::string, std::vector<SearchHit>> buckets;
    for (auto& h : hits) buckets[chain_of(h.slug)].push_back(std::move(h));
    std::vector<SearchHit> out;
    for (auto& [_, lst] : buckets) {
        std::sort(lst.begin(), lst.end(),
                  [](const SearchHit& a, const SearchHit& b) { return a.score > b.score; });
        const auto end = std::min(per_chain, lst.size());
        for (std::size_t i = 0; i < end; ++i) out.push_back(std::move(lst[i]));
    }
    std::sort(out.begin(), out.end(),
              [](const SearchHit& a, const SearchHit& b) { return a.score > b.score; });
    return out;
}

// ── Script detection ────────────────────────────────────────────────

inline bool is_cjk(std::uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF)
        || (cp >= 0x3400 && cp <= 0x4DBF)
        || (cp >= 0x3040 && cp <= 0x30FF)
        || (cp >= 0xAC00 && cp <= 0xD7AF);
}

inline bool is_emoji(std::uint32_t cp) {
    return (cp >= 0x2600 && cp <= 0x27BF) || (cp >= 0x1F300 && cp <= 0x1FAFF);
}

// Iterate codepoints in a UTF-8 string. Callable with a lambda `void(uint32_t)`.
template <class F>
inline void for_each_codepoint(const std::string& s, F&& f) {
    std::size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        std::uint32_t cp = 0;
        std::size_t len = 0;
        if      ((c & 0x80) == 0x00) { cp = c; len = 1; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
        else { ++i; continue; }
        if (i + len > s.size()) break;
        for (std::size_t k = 1; k < len; ++k) {
            cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
        }
        f(cp);
        i += len;
    }
}

inline bool has_cjk(const std::string& s) {
    bool found = false;
    for_each_codepoint(s, [&](std::uint32_t cp) { if (is_cjk(cp)) found = true; });
    return found;
}

inline bool has_emoji(const std::string& s) {
    bool found = false;
    for_each_codepoint(s, [&](std::uint32_t cp) { if (is_emoji(cp)) found = true; });
    return found;
}

struct ScriptReport {
    std::string primary;
    std::map<std::string, double> fractions;
    bool has_cjk = false;
    bool has_emoji = false;
};

inline std::optional<std::string> classify_cp(std::uint32_t cp) {
    if (is_cjk(cp))   return "cjk";
    if (is_emoji(cp)) return "emoji";
    if ((cp >= 0x0041 && cp <= 0x005A) || (cp >= 0x0061 && cp <= 0x007A) || (cp >= 0x00C0 && cp <= 0x024F))
        return "latin";
    if (cp >= 0x0370 && cp <= 0x03FF) return "greek";
    if (cp >= 0x0400 && cp <= 0x04FF) return "cyrillic";
    if (cp >= 0x0590 && cp <= 0x05FF) return "hebrew";
    if (cp >= 0x0600 && cp <= 0x06FF) return "arabic";
    if (cp >= 0x0900 && cp <= 0x097F) return "devanagari";
    if (cp <= 0x002F
        || (cp >= 0x003A && cp <= 0x0040)
        || (cp >= 0x005B && cp <= 0x0060)
        || (cp >= 0x007B && cp <= 0x007E))
        return std::nullopt;
    return "other";
}

inline ScriptReport detect_script(const std::string& s) {
    static const std::vector<std::string> keys = {
        "latin", "cjk", "emoji", "cyrillic", "arabic", "hebrew", "greek", "devanagari", "other"
    };
    std::map<std::string, std::size_t> counts;
    for (const auto& k : keys) counts[k] = 0;
    std::size_t total = 0;
    for_each_codepoint(s, [&](std::uint32_t cp) {
        if (cp >= 0x0030 && cp <= 0x0039) return;
        if (auto k = classify_cp(cp)) {
            counts[*k] += 1;
            total += 1;
        }
    });
    ScriptReport r;
    r.has_cjk = counts["cjk"] > 0;
    r.has_emoji = counts["emoji"] > 0;
    std::size_t max = 0;
    r.primary = "other";
    for (const auto& k : keys) {
        if (counts[k] > max) { max = counts[k]; r.primary = k; }
    }
    for (const auto& k : keys) {
        r.fractions[k] = total > 0 ? static_cast<double>(counts[k]) / static_cast<double>(total) : 0.0;
    }
    return r;
}

// ── FTS helpers (CJK bigrams, emoji trigrams, websearch parser) ─────

inline std::string utf8_encode(std::uint32_t cp) {
    std::string s;
    if (cp < 0x80) s.push_back(static_cast<char>(cp));
    else if (cp < 0x800) {
        s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return s;
}

inline std::vector<std::string> cjk_bigrams(const std::string& text) {
    std::vector<std::string> out;
    std::vector<std::uint32_t> cjk_buf;
    std::string latin_buf;
    auto flush_cjk = [&]() {
        if (cjk_buf.empty()) return;
        if (cjk_buf.size() == 1) {
            out.push_back(utf8_encode(cjk_buf[0]));
        } else {
            for (std::size_t i = 0; i < cjk_buf.size() - 1; ++i) {
                out.push_back(utf8_encode(cjk_buf[i]) + utf8_encode(cjk_buf[i + 1]));
            }
        }
        cjk_buf.clear();
    };
    auto flush_latin = [&]() {
        if (!latin_buf.empty()) { out.push_back(latin_buf); latin_buf.clear(); }
    };
    for_each_codepoint(text, [&](std::uint32_t cp) {
        if (is_cjk(cp)) {
            flush_latin();
            cjk_buf.push_back(cp);
        } else if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r') {
            flush_cjk();
            flush_latin();
        } else {
            flush_cjk();
            latin_buf += utf8_encode(cp);
        }
    });
    flush_cjk();
    flush_latin();
    return out;
}

inline std::vector<std::string> emoji_trigrams(const std::string& text) {
    std::vector<std::uint32_t> chars;
    for_each_codepoint(text, [&](std::uint32_t cp) { chars.push_back(cp); });
    std::vector<std::string> out;
    for (std::size_t i = 0; i < chars.size(); ++i) {
        if (!is_emoji(chars[i])) continue;
        std::string s = utf8_encode(chars[i]);
        if (i + 1 < chars.size()) s += utf8_encode(chars[i + 1]);
        if (i + 2 < chars.size()) s += utf8_encode(chars[i + 2]);
        out.push_back(s);
    }
    return out;
}

struct ParsedQuery {
    std::vector<std::string> required;
    std::vector<std::string> excluded;
    std::vector<std::vector<std::string>> optional;
    std::vector<std::string> phrases;
};

inline ParsedQuery parse_websearch(const std::string& query) {
    struct Tok { std::string kind; std::string value; };
    std::vector<Tok> toks;
    std::size_t i = 0;
    while (i < query.size()) {
        char c = query[i];
        if (std::isspace(static_cast<unsigned char>(c))) { ++i; continue; }
        if (c == '"') {
            std::size_t j = i + 1;
            while (j < query.size() && query[j] != '"') ++j;
            toks.push_back({"phrase", query.substr(i + 1, j - i - 1)});
            i = (j < query.size()) ? j + 1 : j;
            continue;
        }
        std::size_t j = i;
        while (j < query.size() && !std::isspace(static_cast<unsigned char>(query[j]))) ++j;
        toks.push_back({"word", query.substr(i, j - i)});
        i = j;
    }
    ParsedQuery p;
    std::size_t k = 0;
    while (k < toks.size()) {
        const auto& t = toks[k];
        if (t.kind == "phrase") {
            p.phrases.push_back(t.value);
            p.required.push_back(t.value);
            ++k;
            continue;
        }
        if (k + 1 < toks.size() && toks[k + 1].kind == "word" && toks[k + 1].value == "OR") {
            std::vector<std::string> group{t.value};
            std::size_t j = k + 1;
            while (j + 1 < toks.size() && toks[j].kind == "word" && toks[j].value == "OR") {
                group.push_back(toks[j + 1].value);
                j += 2;
            }
            p.optional.push_back(std::move(group));
            k = j;
            continue;
        }
        if (t.value.size() > 1 && t.value[0] == '-') {
            p.excluded.push_back(t.value.substr(1));
            ++k;
            continue;
        }
        p.required.push_back(t.value);
        ++k;
    }
    return p;
}

inline std::string quote_fts5(const std::string& term) {
    bool simple = !term.empty();
    for (char c : term) if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) { simple = false; break; }
    if (!simple || term.find(' ') != std::string::npos) {
        std::string esc = term;
        std::size_t pos = 0;
        while ((pos = esc.find('"', pos)) != std::string::npos) {
            esc.replace(pos, 1, "\"\"");
            pos += 2;
        }
        return "\"" + esc + "\"";
    }
    return term;
}

inline std::string to_fts5_match(const ParsedQuery& p) {
    std::vector<std::string> parts;
    for (const auto& r : p.required) parts.push_back(quote_fts5(r));
    for (const auto& group : p.optional) {
        std::string alts;
        for (std::size_t i = 0; i < group.size(); ++i) {
            if (i > 0) alts += " OR ";
            alts += quote_fts5(group[i]);
        }
        parts.push_back("(" + alts + ")");
    }
    std::string s;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) s += " AND ";
        s += parts[i];
    }
    for (const auto& e : p.excluded) s += " NOT " + quote_fts5(e);
    // trim
    auto not_ws = [](char c) { return !std::isspace(static_cast<unsigned char>(c)); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_ws));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_ws).base(), s.end());
    return s;
}

// ── Embed registry + MRL ────────────────────────────────────────────

struct EmbeddingModel {
    std::string slug;
    std::size_t dim = 0;
    std::vector<std::size_t> mrl_dims;
    std::string prefix_query;
    std::string prefix_passage;
    std::string family;
};

inline std::unordered_map<std::string, EmbeddingModel>& embed_registry() {
    static std::unordered_map<std::string, EmbeddingModel> reg = [] {
        std::unordered_map<std::string, EmbeddingModel> m;
        m["ollama:nomic-embed-text"] = {"ollama:nomic-embed-text", 768, {128, 256, 512, 768}, "", "", "nomic"};
        m["intfloat/e5-large-v2"]    = {"intfloat/e5-large-v2", 1024, {}, "query: ", "passage: ", "e5"};
        m["openai:text-embedding-3-small"] = {"openai:text-embedding-3-small", 1536, {256, 512, 768, 1024, 1536}, "", "", "openai"};
        m["openai:text-embedding-3-large"] = {"openai:text-embedding-3-large", 3072, {256, 512, 1024, 2048, 3072}, "", "", "openai"};
        return m;
    }();
    return reg;
}

inline void register_embedding_model(EmbeddingModel m) { embed_registry()[m.slug] = std::move(m); }

inline std::optional<EmbeddingModel> get_embedding_model(const std::string& slug) {
    auto it = embed_registry().find(slug);
    if (it == embed_registry().end()) return std::nullopt;
    return it->second;
}

inline std::vector<EmbeddingModel> list_embedding_models() {
    std::vector<EmbeddingModel> out;
    for (auto& [_, m] : embed_registry()) out.push_back(m);
    return out;
}

inline std::string prefix_for(const EmbeddingModel& m, const std::string& task, const std::string& text) {
    if (task == "symmetric" || (m.prefix_query.empty() && m.prefix_passage.empty())) return text;
    if (task == "query") return m.prefix_query + text;
    return m.prefix_passage + text;
}

inline void l2_normalize(std::vector<double>& v) {
    double s = 0;
    for (double x : v) s += x * x;
    const double n = std::sqrt(s);
    if (n == 0.0) return;
    for (auto& x : v) x /= n;
}

inline std::vector<double> mrl_truncate(const std::vector<double>& embedding, std::size_t target) {
    if (target == 0) throw std::invalid_argument("target must be > 0");
    std::vector<double> copy;
    if (target >= embedding.size()) copy = embedding;
    else copy.assign(embedding.begin(), embedding.begin() + static_cast<std::ptrdiff_t>(target));
    l2_normalize(copy);
    return copy;
}

inline std::size_t coarse_dim(const EmbeddingModel& m) {
    if (m.mrl_dims.empty()) return m.dim;
    const double target = static_cast<double>(m.dim) / 8.0;
    for (auto d : m.mrl_dims) if (static_cast<double>(d) >= target) return d;
    return m.mrl_dims.back();
}

// ── Temporal (UUIDv7 bounds) ────────────────────────────────────────

inline std::string v7(std::int64_t epoch_ms, bool ceiling) {
    if (epoch_ms < 0) epoch_ms = 0;
    char buf[13];
    std::snprintf(buf, sizeof(buf), "%012llx", static_cast<unsigned long long>(epoch_ms));
    std::string hex(buf);
    if (hex.size() > 12) hex = hex.substr(hex.size() - 12);
    const std::string th = hex.substr(0, 8);
    const std::string tl = hex.substr(8, 4);
    if (ceiling) return th + "-" + tl + "-7fff-bfff-ffffffffffff";
    return th + "-" + tl + "-7000-8000-000000000000";
}

inline std::string v7_floor(std::int64_t epoch_ms) { return v7(epoch_ms, false); }
inline std::string v7_ceiling(std::int64_t epoch_ms) { return v7(epoch_ms, true); }

// ── Captions ────────────────────────────────────────────────────────

struct CaptionSegment {
    double start_secs = 0.0;
    double end_secs = 0.0;
    std::string text;
    std::string speaker;
};

inline std::string fmt_time(double secs, char ms_sep) {
    const int ms = static_cast<int>(std::floor((secs - std::floor(secs)) * 1000.0));
    const int total = static_cast<int>(std::floor(secs));
    const int h = total / 3600;
    const int m = (total % 3600) / 60;
    const int s = total % 60;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02d:%02d:%02d%c%03d", h, m, s, ms_sep, ms);
    return std::string(buf);
}

inline std::string render_vtt(const std::vector<CaptionSegment>& segs) {
    std::string s = "WEBVTT\n\n";
    for (std::size_t i = 0; i < segs.size(); ++i) {
        const auto& sg = segs[i];
        s += std::to_string(i + 1) + "\n";
        s += fmt_time(sg.start_secs, '.') + " --> " + fmt_time(sg.end_secs, '.') + "\n";
        if (!sg.speaker.empty()) s += "<v " + sg.speaker + ">" + sg.text + "</v>\n\n";
        else s += sg.text + "\n\n";
    }
    return s;
}

inline std::string render_srt(const std::vector<CaptionSegment>& segs) {
    std::string s;
    for (std::size_t i = 0; i < segs.size(); ++i) {
        const auto& sg = segs[i];
        s += std::to_string(i + 1) + "\n";
        s += fmt_time(sg.start_secs, ',') + " --> " + fmt_time(sg.end_secs, ',') + "\n";
        if (!sg.speaker.empty()) s += sg.speaker + ": " + sg.text + "\n\n";
        else s += sg.text + "\n\n";
    }
    return s;
}

inline std::string render_rttm(const std::vector<CaptionSegment>& segs, const std::string& uri_in = "audio") {
    std::string uri = uri_in.empty() ? "audio" : uri_in;
    std::string s;
    for (const auto& sg : segs) {
        if (sg.speaker.empty()) continue;
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "SPEAKER %s 1 %.3f %.3f <NA> <NA> %s <NA> <NA>\n",
                      uri.c_str(), sg.start_secs, sg.end_secs - sg.start_secs, sg.speaker.c_str());
        s += buf;
    }
    return s;
}

// ── Tokenizer ───────────────────────────────────────────────────────

inline std::size_t estimate_tokens(const std::string& text) {
    std::size_t total = 0;
    std::string ascii_run;
    auto flush = [&]() {
        if (ascii_run.empty()) return;
        std::istringstream iss(ascii_run);
        std::string w;
        while (iss >> w) {
            std::size_t t = (w.size() + 3) / 4;
            if (t < 1) t = 1;
            total += t;
        }
        ascii_run.clear();
    };
    for_each_codepoint(text, [&](std::uint32_t cp) {
        if (is_cjk(cp) || is_emoji(cp)) { flush(); ++total; }
        else ascii_run += utf8_encode(cp);
    });
    flush();
    return total;
}

inline std::string truncate_to_tokens(const std::string& text, std::size_t max_tokens) {
    if (estimate_tokens(text) <= max_tokens) return text;
    // Walk UTF-8 char boundaries to do a safe binary search.
    std::vector<std::size_t> boundaries;
    boundaries.push_back(0);
    std::size_t i = 0;
    while (i < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        std::size_t len = (c & 0x80) == 0x00 ? 1
                        : (c & 0xE0) == 0xC0 ? 2
                        : (c & 0xF0) == 0xE0 ? 3
                        : (c & 0xF8) == 0xF0 ? 4 : 1;
        i = std::min(i + len, text.size());
        boundaries.push_back(i);
    }
    std::size_t lo = 0, hi = boundaries.size() - 1;
    while (lo < hi) {
        const std::size_t mid = (lo + hi + 1) / 2;
        if (estimate_tokens(text.substr(0, boundaries[mid])) <= max_tokens) lo = mid;
        else hi = mid - 1;
    }
    return text.substr(0, boundaries[lo]);
}

// ── Eval ────────────────────────────────────────────────────────────

struct QueryEval {
    std::vector<std::string> predicted;
    std::unordered_map<std::string, int> relevant;
};

inline std::unordered_set<std::string> rel_set(const QueryEval& q) {
    std::unordered_set<std::string> r;
    for (const auto& [k, v] : q.relevant) if (v > 0) r.insert(k);
    return r;
}

inline double reciprocal_rank(const QueryEval& q) {
    auto r = rel_set(q);
    for (std::size_t i = 0; i < q.predicted.size(); ++i) {
        if (r.count(q.predicted[i])) return 1.0 / static_cast<double>(i + 1);
    }
    return 0.0;
}

inline double mean_reciprocal_rank(const std::vector<QueryEval>& qs) {
    if (qs.empty()) return 0.0;
    double s = 0;
    for (const auto& q : qs) s += reciprocal_rank(q);
    return s / static_cast<double>(qs.size());
}

inline double recall_at_k(const QueryEval& q, std::size_t k) {
    auto r = rel_set(q);
    if (r.empty()) return 0.0;
    std::size_t hits = 0;
    for (std::size_t i = 0; i < std::min(k, q.predicted.size()); ++i) if (r.count(q.predicted[i])) ++hits;
    return static_cast<double>(hits) / static_cast<double>(r.size());
}

inline double precision_at_k(const QueryEval& q, std::size_t k) {
    const std::size_t head = std::min(k, q.predicted.size());
    if (head == 0) return 0.0;
    auto r = rel_set(q);
    std::size_t hits = 0;
    for (std::size_t i = 0; i < head; ++i) if (r.count(q.predicted[i])) ++hits;
    return static_cast<double>(hits) / static_cast<double>(head);
}

inline double ndcg_at_k(const QueryEval& q, std::size_t k) {
    auto dcg = [&](const std::vector<std::string>& slugs) {
        double s = 0;
        for (std::size_t i = 0; i < slugs.size(); ++i) {
            int g = 0;
            auto it = q.relevant.find(slugs[i]);
            if (it != q.relevant.end()) g = it->second;
            s += (std::pow(2.0, static_cast<double>(g)) - 1.0) / std::log2(static_cast<double>(i + 2));
        }
        return s;
    };
    std::vector<std::pair<std::string, int>> sorted(q.relevant.begin(), q.relevant.end());
    std::sort(sorted.begin(), sorted.end(),
              [](auto& a, auto& b) { return a.second > b.second; });
    std::vector<std::string> ideal;
    for (std::size_t i = 0; i < std::min(k, sorted.size()); ++i) ideal.push_back(sorted[i].first);
    const double idcg = dcg(ideal);
    std::vector<std::string> head(q.predicted.begin(),
                                  q.predicted.begin() + static_cast<std::ptrdiff_t>(std::min(k, q.predicted.size())));
    const double actual = dcg(head);
    return idcg > 0 ? actual / idcg : 0.0;
}

// ── Spatial ─────────────────────────────────────────────────────────

constexpr double EARTH_KM = 6371.0088;

inline double deg_to_rad(double d) { return d * M_PI / 180.0; }

inline double haversine_km(double lat_a, double lng_a, double lat_b, double lng_b) {
    const double d_lat = deg_to_rad(lat_b - lat_a);
    const double d_lng = deg_to_rad(lng_b - lng_a);
    const double r1 = deg_to_rad(lat_a);
    const double r2 = deg_to_rad(lat_b);
    const double x = std::pow(std::sin(d_lat / 2), 2)
                   + std::pow(std::sin(d_lng / 2), 2) * std::cos(r1) * std::cos(r2);
    return 2.0 * EARTH_KM * std::asin(std::sqrt(x));
}

struct BBox { double min_lat, min_lng, max_lat, max_lng; };

inline BBox bbox_around(double lat, double lng, double radius_km) {
    const double d_lat = radius_km / 111.0;
    const double d_lng = radius_km / (111.0 * std::cos(deg_to_rad(lat)));
    return {lat - d_lat, lng - d_lng, lat + d_lat, lng + d_lng};
}

inline bool in_box(double lat, double lng, const BBox& b) {
    return lat >= b.min_lat && lat <= b.max_lat && lng >= b.min_lng && lng <= b.max_lng;
}

// ── HTTP Range ──────────────────────────────────────────────────────

struct RangeRequest { std::int64_t start; std::int64_t end; };

enum class RangeOutcome { Ok, Unsatisfiable, Invalid };

inline RangeOutcome parse_range(const std::string& header, std::int64_t total, RangeRequest& out) {
    constexpr const char* prefix = "bytes=";
    if (header.rfind(prefix, 0) != 0) return RangeOutcome::Invalid;
    std::string spec = header.substr(6);
    const auto comma = spec.find(',');
    if (comma != std::string::npos) spec = spec.substr(0, comma);
    // trim
    auto not_ws = [](char c) { return !std::isspace(static_cast<unsigned char>(c)); };
    spec.erase(spec.begin(), std::find_if(spec.begin(), spec.end(), not_ws));
    spec.erase(std::find_if(spec.rbegin(), spec.rend(), not_ws).base(), spec.end());
    if (spec.empty()) return RangeOutcome::Invalid;
    if (spec[0] == '-') {
        try {
            const std::int64_t suffix = std::stoll(spec.substr(1));
            if (suffix <= 0) return RangeOutcome::Invalid;
            out.start = std::max<std::int64_t>(0, total - suffix);
            out.end = total - 1;
            return RangeOutcome::Ok;
        } catch (...) { return RangeOutcome::Invalid; }
    }
    const auto dash = spec.find('-');
    if (dash == std::string::npos) return RangeOutcome::Invalid;
    try {
        out.start = std::stoll(spec.substr(0, dash));
        out.end = (dash + 1 < spec.size())
                  ? std::stoll(spec.substr(dash + 1))
                  : total - 1;
    } catch (...) { return RangeOutcome::Invalid; }
    if (out.start > out.end || out.start >= total) return RangeOutcome::Unsatisfiable;
    if (out.end >= total) out.end = total - 1;
    return RangeOutcome::Ok;
}

inline std::string content_range(std::int64_t start, std::int64_t end, std::int64_t total) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "bytes %lld-%lld/%lld",
                  static_cast<long long>(start),
                  static_cast<long long>(end),
                  static_cast<long long>(total));
    return std::string(buf);
}

// ── Wallet address (base58check) ────────────────────────────────────
//
// BLAKE3 — wallet addresses are content-addressable; every brain runtime
// hashes with BLAKE3 so the output is byte-equivalent across all five
// (TS @noble/hashes/blake3, Python blake3 pip, Go luxfi/crypto/blake3,
// Rust blake3 crate, C++ this header). The C source is pulled at CMake
// configure time from `luxcpp/blake3-reference` — no vendored copy.

extern "C" {
#include "blake3.h"
}

inline constexpr const char* BASE58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
inline constexpr std::uint8_t VERSION_V1 = 0x01;

inline std::array<std::uint8_t, 32> digest32(const std::uint8_t* data, std::size_t len) {
    blake3_hasher h;
    blake3_hasher_init(&h);
    blake3_hasher_update(&h, data, len);
    std::array<std::uint8_t, 32> out{};
    blake3_hasher_finalize(&h, out.data(), out.size());
    return out;
}

inline std::array<std::uint8_t, 32> digest32(const std::vector<std::uint8_t>& v) {
    return digest32(v.data(), v.size());
}

inline std::string base58_encode(const std::vector<std::uint8_t>& bytes) {
    if (bytes.empty()) return "";
    std::size_t zeros = 0;
    while (zeros < bytes.size() && bytes[zeros] == 0) ++zeros;
    std::vector<std::uint8_t> b = bytes;
    std::string out;
    while (!b.empty()) {
        std::uint32_t rem = 0;
        std::vector<std::uint8_t> nx;
        nx.reserve(b.size());
        bool started = false;
        for (auto x : b) {
            std::uint32_t cur = rem * 256u + x;
            std::uint32_t q = cur / 58u;
            rem = cur % 58u;
            if (started || q > 0) { nx.push_back(static_cast<std::uint8_t>(q)); started = true; }
        }
        out.insert(out.begin(), BASE58[rem]);
        b = std::move(nx);
    }
    out.insert(out.begin(), zeros, BASE58[0]);
    return out;
}

inline std::vector<std::uint8_t> base58_decode(const std::string& s) {
    if (s.empty()) return {};
    std::size_t zeros = 0;
    while (zeros < s.size() && s[zeros] == BASE58[0]) ++zeros;
    std::vector<std::uint8_t> n;
    n.push_back(0);
    for (char c : s) {
        const auto pos = std::string(BASE58).find(c);
        if (pos == std::string::npos) throw std::invalid_argument("base58: invalid char");
        std::uint32_t carry = static_cast<std::uint32_t>(pos);
        for (auto& byte : n) {
            carry += static_cast<std::uint32_t>(byte) * 58u;
            byte = static_cast<std::uint8_t>(carry & 0xff);
            carry >>= 8;
        }
        while (carry > 0) { n.push_back(static_cast<std::uint8_t>(carry & 0xff)); carry >>= 8; }
    }
    std::vector<std::uint8_t> out(zeros, 0);
    out.insert(out.end(), n.rbegin(), n.rend());
    // Strip leading zero introduced by the seed-byte
    while (out.size() > zeros && out[zeros] == 0) out.erase(out.begin() + static_cast<std::ptrdiff_t>(zeros));
    return out;
}

inline std::string encode_address(const std::vector<std::uint8_t>& public_key,
                                  const std::string& prefix = "hanzo") {
    if (public_key.size() != 32) throw std::invalid_argument("public key must be 32 bytes");
    auto h = digest32(public_key);
    std::vector<std::uint8_t> versioned;
    versioned.reserve(21);
    versioned.push_back(VERSION_V1);
    versioned.insert(versioned.end(), h.begin(), h.begin() + 20);
    auto cs = digest32(versioned);
    std::vector<std::uint8_t> payload = versioned;
    payload.insert(payload.end(), cs.begin(), cs.begin() + 4);
    return prefix + ":" + base58_encode(payload);
}

struct DecodedAddress {
    std::string prefix;
    std::uint8_t version;
    std::array<std::uint8_t, 20> hash;
};

inline DecodedAddress decode_address(const std::string& addr) {
    auto colon = addr.find(':');
    if (colon == std::string::npos) throw std::invalid_argument("address: missing prefix");
    auto decoded = base58_decode(addr.substr(colon + 1));
    if (decoded.size() != 25) throw std::invalid_argument("address: wrong length");
    DecodedAddress out;
    out.prefix = addr.substr(0, colon);
    out.version = decoded[0];
    std::copy(decoded.begin() + 1, decoded.begin() + 21, out.hash.begin());
    auto expected = digest32(std::vector<std::uint8_t>(decoded.begin(), decoded.begin() + 21));
    for (std::size_t i = 0; i < 4; ++i) {
        if (decoded[21 + i] != expected[i]) throw std::invalid_argument("address: bad checksum");
    }
    return out;
}

// ── Graph maintenance ───────────────────────────────────────────────

struct WeightedEdge { std::string source; std::string target; double weight; };

inline std::vector<WeightedEdge> normalize_edges(const std::vector<WeightedEdge>& edges) {
    if (edges.empty()) return {};
    double lo = std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    for (const auto& e : edges) { if (e.weight < lo) lo = e.weight; if (e.weight > hi) hi = e.weight; }
    const double span = hi - lo;
    std::vector<WeightedEdge> out;
    out.reserve(edges.size());
    for (const auto& e : edges) {
        out.push_back({e.source, e.target, span > 0 ? (e.weight - lo) / span : 1.0});
    }
    return out;
}

inline std::vector<WeightedEdge> snn_score(const std::vector<WeightedEdge>& edges, std::size_t k) {
    std::unordered_map<std::string, std::vector<std::pair<std::string, double>>> adj;
    for (const auto& e : edges) {
        adj[e.source].push_back({e.target, e.weight});
        adj[e.target].push_back({e.source, e.weight});
    }
    std::unordered_map<std::string, std::unordered_set<std::string>> nbrs;
    for (auto& [node, lst] : adj) {
        std::sort(lst.begin(), lst.end(),
                  [](auto& a, auto& b) { return a.second > b.second; });
        std::unordered_set<std::string> set;
        for (std::size_t i = 0; i < std::min(k, lst.size()); ++i) set.insert(lst[i].first);
        nbrs[node] = std::move(set);
    }
    std::vector<WeightedEdge> out;
    out.reserve(edges.size());
    for (const auto& e : edges) {
        const auto& a = nbrs[e.source];
        const auto& b = nbrs[e.target];
        std::size_t inter = 0;
        for (const auto& n : a) if (b.count(n)) ++inter;
        const std::size_t uni = a.size() + b.size() - inter;
        out.push_back({e.source, e.target, uni > 0 ? static_cast<double>(inter) / static_cast<double>(uni) : 0.0});
    }
    return out;
}

inline std::vector<WeightedEdge> pfnet_infinity(const std::vector<WeightedEdge>& edges) {
    std::unordered_map<std::string, std::unordered_map<std::string, double>> adj;
    for (const auto& e : edges) {
        auto& m = adj[e.source];
        auto it = m.find(e.target);
        if (it == m.end() || e.weight > it->second) m[e.target] = e.weight;
    }
    std::vector<WeightedEdge> keep;
    for (const auto& e : edges) {
        bool dominated = false;
        auto srcs = adj.find(e.source);
        if (srcs != adj.end()) {
            for (const auto& [x, w_ux] : srcs->second) {
                if (x == e.target) continue;
                auto a2 = adj.find(x);
                if (a2 == adj.end()) continue;
                auto w_xv_it = a2->second.find(e.target);
                if (w_xv_it == a2->second.end()) continue;
                if (std::min(w_ux, w_xv_it->second) > e.weight) { dominated = true; break; }
            }
        }
        if (!dominated) keep.push_back(e);
    }
    return keep;
}

inline std::unordered_map<std::string, int> louvain(const std::vector<WeightedEdge>& edges, std::size_t passes = 10) {
    std::unordered_set<std::string> nodes;
    for (const auto& e : edges) { nodes.insert(e.source); nodes.insert(e.target); }
    std::unordered_map<std::string, int> community;
    {
        int i = 0;
        for (const auto& n : nodes) community[n] = i++;
    }
    std::unordered_map<std::string, std::vector<std::pair<std::string, double>>> adj;
    double total = 0;
    for (const auto& e : edges) {
        adj[e.source].push_back({e.target, e.weight});
        adj[e.target].push_back({e.source, e.weight});
        total += e.weight;
    }
    std::unordered_map<std::string, double> deg;
    for (const auto& [n, lst] : adj) {
        double d = 0; for (const auto& [_, w] : lst) d += w;
        deg[n] = d;
    }
    const double m = total;
    for (std::size_t p = 0; p < passes; ++p) {
        bool improved = false;
        for (const auto& n : nodes) {
            const int cur = community[n];
            std::unordered_map<int, double> w_to;
            for (const auto& [nb, w] : adj[n]) w_to[community[nb]] += w;
            int best = cur;
            double best_gain = 0;
            const double kn = deg[n];
            for (const auto& [c, wnc] : w_to) {
                if (c == cur) continue;
                double sigma = 0;
                for (const auto& [other, comm] : community) {
                    if (comm == c && other != n) sigma += deg[other];
                }
                const double gain = wnc - (kn * sigma) / std::max(2.0 * m, 1e-9);
                if (gain > best_gain) { best_gain = gain; best = c; }
            }
            if (best != cur) { community[n] = best; improved = true; }
        }
        if (!improved) break;
    }
    std::unordered_map<int, int> id_map;
    int next = 0;
    for (const auto& [_, c] : community) {
        if (!id_map.count(c)) id_map[c] = next++;
    }
    for (auto& [_, c] : community) c = id_map[c];
    return community;
}

// ── Inference: provider slug + runtime config + link types ──────────

inline bool is_known_provider(const std::string& s) {
    static const std::set<std::string> known = {
        "ollama", "openai", "openrouter", "llamacpp",
        "anthropic", "google", "azure", "groq", "together", "mock"
    };
    return known.count(s) > 0;
}

struct ParsedSlug { std::string provider; std::string model; };

inline ParsedSlug parse_slug(const std::string& slug, const std::string& default_provider = "ollama") {
    const auto colon = slug.find(':');
    if (colon == std::string::npos) return {default_provider, slug};
    const auto head = slug.substr(0, colon);
    if (is_known_provider(head)) return {head, slug.substr(colon + 1)};
    return {default_provider, slug};
}

inline std::string format_slug(const ParsedSlug& p) { return p.provider + ":" + p.model; }

class RuntimeConfig {
public:
    std::unordered_map<std::string, std::string> defaults;
    std::unordered_map<std::string, std::string> env;

    RuntimeConfig() = default;
    RuntimeConfig(std::unordered_map<std::string, std::string> d,
                  std::unordered_map<std::string, std::string> e)
        : defaults(std::move(d)), env(std::move(e)) {}

    std::optional<std::string> get(const std::string& key) const {
        auto o = overrides.find(key);
        if (o != overrides.end()) return o->second;
        auto e = env.find(key);
        if (e != env.end()) return e->second;
        auto d = defaults.find(key);
        if (d != defaults.end()) return d->second;
        return std::nullopt;
    }
    std::string source(const std::string& key) const {
        if (overrides.count(key)) return "db_override";
        if (env.count(key)) return "env";
        if (defaults.count(key)) return "default";
        return "absent";
    }
    void set(const std::string& key, const std::string& value) { overrides[key] = value; }
    void clear(const std::string& key) { overrides.erase(key); }

private:
    std::unordered_map<std::string, std::string> overrides;
};

inline std::string classify_link_rule(const std::string& evidence) {
    std::string e = evidence;
    std::transform(e.begin(), e.end(), e.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (e.find("founded") != std::string::npos) return "founded";
    if (e.find("invested") != std::string::npos && e.find("in") != std::string::npos) return "invested_in";
    if (e.find("advisor") != std::string::npos || e.find("advises") != std::string::npos || e.find("advising") != std::string::npos) return "advises";
    if (e.find("works at") != std::string::npos || e.find("works for") != std::string::npos
        || e.find("work at") != std::string::npos || e.find("work for") != std::string::npos) return "works_at";
    if (e.find("attended") != std::string::npos) return "attended";
    if (e.find("wrote") != std::string::npos || e.find("authored") != std::string::npos) return "authored";
    if (e.find("cites") != std::string::npos || e.find("cite ") != std::string::npos) return "cites";
    if (e.find("succeeded by") != std::string::npos) return "succeeded_by";
    if (e.find("located in") != std::string::npos) return "located_in";
    return "mentions";
}

// ── Circuit breaker + retry ─────────────────────────────────────────

class CircuitOpenError : public std::runtime_error {
public:
    CircuitOpenError() : std::runtime_error("circuit open") {}
};

class CircuitBreaker {
public:
    CircuitBreaker(std::size_t failure_threshold = 5,
                   std::int64_t cooldown_ms = 30000)
        : threshold_(failure_threshold), cooldown_ms_(cooldown_ms) {}

    enum class State { Closed, Open, HalfOpen };

    State state() const {
        if (failures_ < threshold_) return State::Closed;
        if (now_ms() - opened_at_ >= cooldown_ms_) return State::HalfOpen;
        return State::Open;
    }

    template <class F>
    auto run(F&& fn) -> decltype(fn()) {
        if (state() == State::Open) throw CircuitOpenError();
        try {
            auto r = fn();
            failures_ = 0; opened_at_ = 0;
            return r;
        } catch (...) {
            ++failures_;
            if (failures_ >= threshold_ && opened_at_ == 0) opened_at_ = now_ms();
            throw;
        }
    }

private:
    std::size_t threshold_;
    std::int64_t cooldown_ms_;
    std::size_t failures_ = 0;
    std::int64_t opened_at_ = 0;

    static std::int64_t now_ms() {
        using namespace std::chrono;
        return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
    }
};

} // namespace hanzo::brain
