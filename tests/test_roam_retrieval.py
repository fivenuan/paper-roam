import unittest

from roam_retrieval import Document, RetrievalIndex, SearchHit, build_context


class RetrievalTests(unittest.TestCase):
    def test_bm25_finds_chinese_phrase_in_longer_text(self):
        index = RetrievalIndex([
            Document("a", "偏好和兴趣变化"),
            Document("b", "助手的长期记忆能保存偏好更新吗？"),
        ])
        hits = index.rank("偏好更新")
        self.assertEqual([hit.document.id for hit in hits], ["b", "a"])
        self.assertGreater(hits[0].score, hits[1].score)

    def test_bm25_normalizes_text_and_omits_zero_overlap(self):
        index = RetrievalIndex([
            Document("a", "ＭＥＭＯＲＹ"),
            Document("b", "forecasting"),
        ])
        self.assertEqual([hit.document.id for hit in index.rank("memory")], ["a"])
        self.assertEqual(index.rank("unrelated"), [])

    def test_hybrid_adds_semantic_candidate_and_reuses_document_embeddings(self):
        calls = []

        def embed(texts):
            vectors = {
                "paper preferences": [1, 0],
                "diachronic intent": [10, 10],
                "preferences": [1, 1],
            }
            calls.append(list(texts))
            return [vectors[text] for text in texts]

        index = RetrievalIndex([
            Document("lexical", "paper preferences"),
            Document("semantic", "diachronic intent"),
        ], embed=embed)
        self.assertEqual([hit.document.id for hit in index.rank("preferences")], ["lexical"])
        self.assertEqual(calls, [])
        for _ in range(2):
            hits = index.rank("preferences", method="hybrid")
            self.assertEqual([hit.document.id for hit in hits], ["lexical", "semantic"])
            self.assertAlmostEqual(hits[0].score, 1 / 61 + 1 / 62)
            self.assertAlmostEqual(hits[1].score, 1 / 61)
        self.assertEqual(calls, [
            ["paper preferences", "diachronic intent"], ["preferences"], ["preferences"]
        ])

    def test_reranker_can_promote_below_the_output_limit(self):
        index = RetrievalIndex([
            Document("a", "research memory"),
            Document("b", "seminar notes"),
        ], embed=lambda texts: [[1] for _ in texts])
        hits = index.rank(
            "memory",
            method="reranked",
            limit=1,
            rerank=lambda query, documents: [float(doc.id == "b") for doc in documents],
        )
        self.assertEqual([hit.document.id for hit in hits], ["b"])

    def test_unavailable_methods_are_not_silently_downgraded(self):
        documents = [Document("a", "memory")]
        with self.assertRaisesRegex(ValueError, "embedding"):
            RetrievalIndex(documents).rank("memory", method="hybrid")
        with self.assertRaisesRegex(ValueError, "reranker"):
            RetrievalIndex(documents, embed=lambda texts: [[1]]).rank("memory", method="reranked")

    def test_provider_wrong_vector_dimensions_are_rejected(self):
        def embed(texts):
            return [[1, 0] if text == "memory" else [1] for text in texts]

        index = RetrievalIndex([Document("a", "memory")], embed=embed)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            index.rank("query", method="hybrid")

    def test_context_preserves_provenance_and_character_budget(self):
        hits = [
            SearchHit(Document("a", "ten characters", "Paper §2"), 1),
            SearchHit(Document("b", "next excerpt", "Paper §3"), 0.5),
        ]
        self.assertEqual(build_context(hits), "[a] Paper §2\nten characters\n\n[b] Paper §3\nnext excerpt")
        self.assertEqual(build_context(hits, max_chars=17), "[a] Paper §2\nten…")
        self.assertEqual(build_context(hits, max_chars=3), "")
