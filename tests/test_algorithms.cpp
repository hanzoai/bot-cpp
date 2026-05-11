// tests/test_algorithms.cpp — C++ parity test runner.

#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>

#include "hanzo/brain/algorithms.hpp"

using namespace hanzo::brain;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond) do { \
    if (cond) { ++g_pass; } \
    else { ++g_fail; std::cerr << "FAIL: " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; } \
} while (0)

#define CHECK_EQ(a, b) do { \
    auto _a = (a); auto _b = (b); \
    if (_a == _b) { ++g_pass; } \
    else { ++g_fail; std::cerr << "FAIL: " << __FILE__ << ":" << __LINE__ << ": got=" << _a << " want=" << _b << "\n"; } \
} while (0)

#define CHECK_NEAR(a, b, eps) do { \
    double _a = (a); double _b = (b); \
    if (std::fabs(_a - _b) <= (eps)) { ++g_pass; } \
    else { ++g_fail; std::cerr << "FAIL: " << __FILE__ << ":" << __LINE__ << ": got=" << _a << " want=" << _b << "\n"; } \
} while (0)

static SearchHit hit(std::string slug, double score) {
    return {std::move(slug), score, slug, "keyword"};
}

int main() {
    // ── Fusion ────────────────────────────────────────────────────────
    {
        auto r = rrf_fuse({{hit("a", 1.0), hit("b", 0.5)}}, 10);
        CHECK_EQ(r[0].slug, "a");
        CHECK_NEAR(r[0].score, 1.0, 0.01);
    }
    {
        auto r = rrf_fuse({{hit("a", 1.0)}, {hit("a", 1.0), hit("b", 0.5)}}, 10);
        CHECK_EQ(r[0].slug, "a");
    }
    {
        auto r = rsf_fuse({{hit("a", 100.0), hit("b", 50.0)}, {hit("a", 1.0), hit("c", 0.5)}}, 10);
        CHECK_EQ(r[0].slug, "a");
        CHECK_EQ(r.size(), 3u);
    }
    CHECK_EQ(select_rrf_k(characterize("\"hello world\"")), 10);
    CHECK_EQ(select_rrf_k(characterize("foo AND bar")), 15);
    CHECK_EQ(select_rrf_k(characterize("rust")), 15);
    CHECK_EQ(select_rrf_k(characterize("a b c d e f g h i j")), 40);
    {
        auto sw = select_weights(characterize("rust"));
        CHECK(sw.fts > sw.semantic);
        auto lw = select_weights(characterize("how do retrieval augmented generation systems typically work in production scale"));
        CHECK(lw.semantic > lw.fts);
    }

    // ── Rerank (MMR) ──────────────────────────────────────────────────
    CHECK_NEAR(cosine({1, 0}, {1, 0}), 1.0, 1e-6);
    CHECK_NEAR(cosine({1, 0}, {0, 1}), 0.0, 1e-6);
    {
        std::vector<MmrInput> hits = {
            {hit("a", 0.9),  std::vector<double>{1.0, 0.0}},
            {hit("b", 0.85), std::vector<double>{1.0, 0.01}},
            {hit("c", 0.6),  std::vector<double>{0.0, 1.0}},
        };
        auto out = mmr_rerank(hits, 0.2, 2);
        CHECK_EQ(out[0].hit.slug, "a");
        CHECK_EQ(out[1].hit.slug, "c");
    }

    // ── Dedup ─────────────────────────────────────────────────────────
    {
        auto out = dedup_hits({
            hit("page/foo#chunk-0", 0.5),
            hit("page/foo#chunk-1", 0.8),
            hit("page/bar", 0.6),
        }, 1);
        CHECK_EQ(out.size(), 2u);
    }

    // ── Script detection ──────────────────────────────────────────────
    CHECK_EQ(detect_script("こんにちは世界").primary, std::string("cjk"));
    CHECK_EQ(detect_script("Hello world").primary, std::string("latin"));
    CHECK_EQ(detect_script("Привет").primary, std::string("cyrillic"));
    CHECK(has_cjk("こんにちは"));
    CHECK(!has_cjk("hello"));
    CHECK(has_emoji("hi 🚀"));

    // ── FTS helpers ───────────────────────────────────────────────────
    {
        auto out = cjk_bigrams("hello 世界 こんにちは");
        bool has_hello = false, has_shijie = false, has_kon = false;
        for (auto& t : out) {
            if (t == "hello") has_hello = true;
            if (t == "世界") has_shijie = true;
            if (t == "こん") has_kon = true;
        }
        CHECK(has_hello);
        CHECK(has_shijie);
        CHECK(has_kon);
    }
    {
        auto out = emoji_trigrams("hi 🚀🌌🌟");
        CHECK(!out.empty());
    }
    {
        auto p = parse_websearch("\"hello world\" foo OR bar -baz qux");
        CHECK_EQ(p.phrases.size(), 1u);
        CHECK_EQ(p.phrases[0], std::string("hello world"));
        CHECK_EQ(p.optional.size(), 1u);
        CHECK_EQ(p.optional[0].size(), 2u);
        CHECK_EQ(p.optional[0][0], std::string("foo"));
        CHECK_EQ(p.optional[0][1], std::string("bar"));
        CHECK_EQ(p.excluded.size(), 1u);
        CHECK_EQ(p.excluded[0], std::string("baz"));
    }
    {
        auto sql = to_fts5_match(parse_websearch("apple OR orange -spoil"));
        CHECK(sql.find("apple OR orange") != std::string::npos);
        CHECK(sql.find("NOT spoil") != std::string::npos);
    }

    // ── Embed registry + MRL ──────────────────────────────────────────
    CHECK_EQ(get_embedding_model("ollama:nomic-embed-text")->dim, 768u);
    CHECK_EQ(get_embedding_model("openai:text-embedding-3-small")->dim, 1536u);
    {
        auto e5 = *get_embedding_model("intfloat/e5-large-v2");
        CHECK_EQ(prefix_for(e5, "query", "x"), std::string("query: x"));
        CHECK_EQ(prefix_for(e5, "passage", "x"), std::string("passage: x"));
        auto nomic = *get_embedding_model("ollama:nomic-embed-text");
        CHECK_EQ(prefix_for(nomic, "query", "x"), std::string("x"));
    }
    {
        std::vector<double> v = {1, 2, 3, 4, 5, 6, 7, 8};
        auto t = mrl_truncate(v, 4);
        CHECK_EQ(t.size(), 4u);
        double n = 0;
        for (double x : t) n += x * x;
        CHECK_NEAR(std::sqrt(n), 1.0, 1e-6);
    }
    {
        auto m = *get_embedding_model("openai:text-embedding-3-large");
        auto cd = coarse_dim(m);
        CHECK(cd >= 256u && cd <= 512u);
    }
    {
        std::vector<double> v = {0, 0, 0};
        l2_normalize(v);
        CHECK_EQ(v[0], 0.0);
    }

    // ── Temporal ──────────────────────────────────────────────────────
    {
        auto t = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        CHECK(v7_floor(t) < v7_ceiling(t));
    }

    // ── Captions ──────────────────────────────────────────────────────
    {
        std::vector<CaptionSegment> segs = {
            {0.0, 1.5, "hi", "S0"},
            {1.5, 3.0, "world", "S1"},
        };
        CHECK_EQ(render_vtt(segs).substr(0, 6), std::string("WEBVTT"));
        CHECK(render_srt(segs).find("00:00:00,000 --> 00:00:01,500") != std::string::npos);
        CHECK_EQ(render_rttm(segs).substr(0, 7), std::string("SPEAKER"));
    }

    // ── Tokenizer ─────────────────────────────────────────────────────
    CHECK(estimate_tokens("hi there friend") > estimate_tokens("hi"));
    CHECK_EQ(estimate_tokens("こんにちは"), 5u);
    {
        std::string longt;
        for (int i = 0; i < 100; ++i) longt += "alpha ";
        auto tr = truncate_to_tokens(longt, 20);
        CHECK(estimate_tokens(tr) <= 20u);
    }

    // ── Eval ──────────────────────────────────────────────────────────
    {
        QueryEval q;
        q.predicted = {"a", "b", "c", "d"};
        q.relevant = {{"c", 1}, {"d", 1}};
        CHECK_NEAR(reciprocal_rank(q), 1.0 / 3.0, 1e-6);
        CHECK_NEAR(recall_at_k(q, 2), 0.0, 1e-6);
        CHECK_NEAR(recall_at_k(q, 4), 1.0, 1e-6);
        CHECK_NEAR(precision_at_k(q, 4), 0.5, 1e-6);

        QueryEval graded;
        graded.predicted = {"a", "b"};
        graded.relevant = {{"a", 3}, {"b", 1}};
        CHECK(ndcg_at_k(graded, 2) > 0.9);

        CHECK(mean_reciprocal_rank({q}) > 0);
    }

    // ── Spatial ───────────────────────────────────────────────────────
    CHECK_NEAR(haversine_km(0, 0, 0, 0), 0.0, 1e-6);
    CHECK(std::fabs(haversine_km(40.7128, -74.006, 34.0522, -118.2437) - 3935.0) < 50.0);
    {
        auto box = bbox_around(37.77, -122.42, 10.0);
        CHECK(in_box(37.77, -122.42, box));
        CHECK(!in_box(0.0, 0.0, box));
    }

    // ── HTTP Range ────────────────────────────────────────────────────
    {
        RangeRequest r{};
        CHECK(parse_range("bytes=0-99", 1000, r) == RangeOutcome::Ok);
        CHECK_EQ(static_cast<int>(r.start), 0); CHECK_EQ(static_cast<int>(r.end), 99);
        CHECK(parse_range("bytes=-100", 1000, r) == RangeOutcome::Ok);
        CHECK_EQ(static_cast<int>(r.start), 900); CHECK_EQ(static_cast<int>(r.end), 999);
        CHECK(parse_range("bytes=2000-3000", 1000, r) == RangeOutcome::Unsatisfiable);
        CHECK_EQ(content_range(0, 99, 1000), std::string("bytes 0-99/1000"));
    }

    // ── Wallet address ────────────────────────────────────────────────
    {
        std::vector<std::uint8_t> pk(32);
        for (std::size_t i = 0; i < 32; ++i) pk[i] = static_cast<std::uint8_t>(i);
        auto addr = encode_address(pk);
        CHECK_EQ(addr.substr(0, 6), std::string("hanzo:"));
        auto dec = decode_address(addr);
        CHECK_EQ(dec.prefix, std::string("hanzo"));
        CHECK_EQ(static_cast<int>(dec.version), 1);

        bool threw = false;
        try { decode_address("hanzo:11111111111111111111111111"); }
        catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);

        auto mm = encode_address(pk, "mm");
        CHECK_EQ(mm.substr(0, 3), std::string("mm:"));
    }

    // ── Graph maintenance ─────────────────────────────────────────────
    {
        auto out = normalize_edges({{"a", "b", 10}, {"b", "c", 5}});
        CHECK_NEAR(out[0].weight, 1.0, 1e-6);
        CHECK_NEAR(out[1].weight, 0.0, 1e-6);
    }
    {
        auto out = snn_score({{"a", "b", 0.9}, {"a", "c", 0.8}, {"b", "c", 0.7}}, 2);
        for (auto& e : out) {
            CHECK(e.weight >= 0.0 && e.weight <= 1.0);
        }
    }
    {
        auto out = pfnet_infinity({{"a", "b", 0.9}, {"b", "c", 0.9}, {"a", "c", 0.5}});
        bool found_ac = false;
        for (auto& e : out) if (e.source == "a" && e.target == "c") found_ac = true;
        CHECK(!found_ac);
    }
    {
        auto c = louvain({{"a", "b", 1}, {"b", "c", 1}, {"a", "c", 1}});
        CHECK_EQ(c.size(), 3u);
    }

    // ── Inference: slug + runtime config + link types ────────────────
    {
        auto p = parse_slug("openai:gpt-4o");
        CHECK_EQ(p.provider, std::string("openai"));
        CHECK_EQ(p.model, std::string("gpt-4o"));
        auto p2 = parse_slug("qwen3:8b");
        CHECK_EQ(p2.provider, std::string("ollama"));
        CHECK_EQ(p2.model, std::string("qwen3:8b"));
        CHECK_EQ(format_slug({"openai", "gpt-4o"}), std::string("openai:gpt-4o"));
    }
    {
        RuntimeConfig rc({{"K", "default"}}, {{"K", "env"}});
        CHECK_EQ(*rc.get("K"), std::string("env"));
        rc.set("K", "override");
        CHECK_EQ(*rc.get("K"), std::string("override"));
        CHECK_EQ(rc.source("K"), std::string("db_override"));
        rc.clear("K");
        CHECK_EQ(*rc.get("K"), std::string("env"));
    }
    CHECK_EQ(classify_link_rule("Alice founded Acme"), std::string("founded"));
    CHECK_EQ(classify_link_rule("Alice invested in Acme"), std::string("invested_in"));
    CHECK_EQ(classify_link_rule("worked together"), std::string("mentions"));

    // ── Circuit breaker ───────────────────────────────────────────────
    {
        CircuitBreaker cb(2, 100);
        auto fail = []() -> int { throw std::runtime_error("nope"); };
        bool threw = false;
        try { cb.run(fail); } catch (...) { threw = true; } CHECK(threw);
        threw = false;
        try { cb.run(fail); } catch (...) { threw = true; } CHECK(threw);
        CHECK(cb.state() == CircuitBreaker::State::Open);
        bool circuit_open = false;
        try { cb.run(fail); } catch (const CircuitOpenError&) { circuit_open = true; } catch (...) {}
        CHECK(circuit_open);
    }

    std::cout << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail == 0 ? 0 : 1;
}
