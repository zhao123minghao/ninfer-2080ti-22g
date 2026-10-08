from tools.artifact.container import (
    Artifact,
    ArtifactIdentity,
    ArtifactWriter,
    ResourceSpec,
    TensorSpec,
)
from tools.convert.qwen3_8_27b.attach_dflash2 import append_dflash2_objects
from tools.convert.qwen3_8_27b.convert_gguf import DirectObject


IDENTITY = ArtifactIdentity("qwen3.8-27b", "groupwise-int")


def make_base_artifact(path):
    specs = [
        ResourceSpec("frontend/test.json", "raw-bytes-v1", 2),
        TensorSpec("text/test", (3,), "BF16", "contiguous-le-v1"),
    ]
    with ArtifactWriter(path, IDENTITY, specs) as writer:
        writer.write("frontend/test.json", b"{}")
        writer.write("text/test", bytes((1, 2, 3, 4, 5, 6)))


def test_append_preserves_identity_descriptors_and_base_payloads(tmp_path):
    source_path = tmp_path / "base.ninfer"
    output_path = tmp_path / "with_dflash.ninfer"
    make_base_artifact(source_path)
    draft_payload = bytes((11, 12, 13, 14))
    draft = DirectObject(
        TensorSpec("dflash/test", (2,), "BF16", "contiguous-le-v1"),
        draft_payload,
    )

    append_dflash2_objects(source_path, output_path, [draft])

    with Artifact.open(source_path) as source, Artifact.open(output_path) as output:
        assert output.identity == source.identity == IDENTITY
        assert output.objects[:len(source.objects)] == source.objects
        for obj in source.objects:
            assert output.payload(obj.name) == source.payload(obj)
        assert output.payload("dflash/test") == draft_payload


def test_append_rejects_overwriting_the_source(tmp_path):
    source_path = tmp_path / "base.ninfer"
    make_base_artifact(source_path)
    draft = DirectObject(
        TensorSpec("dflash/test", (2,), "BF16", "contiguous-le-v1"),
        bytes((11, 12, 13, 14)),
    )

    try:
        append_dflash2_objects(source_path, source_path, [draft])
    except ValueError as error:
        assert "must not replace" in str(error)
    else:
        raise AssertionError("expected source overwrite to be rejected")