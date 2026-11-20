import pytest

from ragbot.chunking import chunk_text


def test_chunks_have_size_and_overlap():
    text = "".join(str(i % 10) for i in range(1200))
    chunks = chunk_text(text, size=500, overlap=100)

    assert len(chunks) == 3                      # starts at 0, 400, 800
    assert all(len(c) <= 500 for c in chunks)
    assert chunks[0][-100:] == chunks[1][:100]   # neighbours share the overlap


def test_every_character_is_covered():
    text = "あいうえお" * 300
    chunks = chunk_text(text, size=500, overlap=100)
    assert chunks[0].startswith(text[:500])
    assert text.endswith(chunks[-1])


def test_short_and_empty_text():
    assert chunk_text("短い文", size=500, overlap=100) == ["短い文"]
    assert chunk_text("", size=500, overlap=100) == []


@pytest.mark.parametrize("size, overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_settings(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("abc", size=size, overlap=overlap)
