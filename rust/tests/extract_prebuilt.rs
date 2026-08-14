//! Positive round-trip integration test for `extract_prebuilt_fullmap`.
//!
//! WHY this test exists: the whole point of distributing a prebuilt
//! `fullmap.tar.zst` is that extracting it must land EXACTLY the database a
//! local force build (`build_fullmap_db`) produces — any divergence silently
//! corrupts every lookup downstream.  This test builds the shared fixture DB
//! with the production build entry point, packages the primary + all 16
//! shards into a zstd-compressed tar (nested in a subdirectory, as real
//! archives may be laid out), extracts it through the PUBLIC
//! `extract_prebuilt_fullmap` pyfunction — including its `py.detach`
//! GIL-release path, driven with the same `Python::initialize` +
//! `Python::attach` pattern `build_golden` uses for `build_fullmap_db` — and
//! pins:
//!   * the primary lands at the requested output path;
//!   * all 16 shards land as `<stem>.s<i>.redb` with no gaps and no extras;
//!   * the dot-prefixed temp dir is fully removed on success;
//!   * `canonical_dump(term_curie_map(...))` of the extracted bundle EQUALS
//!     that of the force-built fixture — the "matches what a force build
//!     produces" equivalence pin, in the exact canonical form `build_golden`
//!     pins (term-sorted lines, CURIE strings sorted within each term, so it
//!     is invariant under thread count and curie_id assignment).
//!
//! Fixtures and canonical-form helpers are shared with `build_golden` via
//! `tests/common/mod.rs` so both tests compare against one definition of the
//! canonical form.  Negative paths (torn archives, schema/build_id mismatches,
//! temp-dir cleanup on failure) are covered by a follow-up story.

mod common;

use std::fs::File;

#[test]
fn extract_prebuilt_matches_force_build() {
    // 1. Build the fixture with the production entry point (single-threaded
    //    for a deterministic, fast build).
    let fixture_dir = tempfile::tempdir().unwrap();
    let fixture_primary = common::build_fixture(fixture_dir.path(), 1);

    // 2. Package the primary + all 16 shards into `fullmap.tar.zst`.  Members
    //    are nested under a `bundle/` subdirectory to exercise the recursive
    //    scan path, and an explicit directory entry exercises the
    //    `EntryType::Directory` extraction branch.
    let archive_path = fixture_dir.path().join("fullmap.tar.zst");
    {
        let archive_file = File::create(&archive_path).unwrap();
        let encoder = zstd::Encoder::new(archive_file, 0).unwrap();
        let mut builder = tar::Builder::new(encoder);

        let mut dir_header = tar::Header::new_gnu();
        dir_header.set_path("bundle/").unwrap();
        dir_header.set_entry_type(tar::EntryType::Directory);
        dir_header.set_size(0);
        dir_header.set_mode(0o755);
        dir_header.set_cksum();
        builder.append(&dir_header, std::io::empty()).unwrap();

        let mut primary_file = File::open(&fixture_primary).unwrap();
        builder
            .append_file("bundle/fullmap.redb", &mut primary_file)
            .unwrap();
        for index in 0..common::SHARD_COUNT {
            let shard = common::shard_path(&fixture_primary, index);
            let member = format!("bundle/{}", shard.file_name().unwrap().to_string_lossy());
            let mut shard_file = File::open(&shard).unwrap();
            builder.append_file(&member, &mut shard_file).unwrap();
        }

        // Flush tar, then finalize the zstd frame so the file is complete.
        let encoder = builder.into_inner().unwrap();
        encoder.finish().unwrap();
    }

    // 3. Extract through the public pyfunction, exactly as Python callers do
    //    (the GIL token is required by the signature; `Python::attach` +
    //    `py.detach` inside the function is the production code path).
    let out_dir = tempfile::tempdir().unwrap();
    let output = out_dir.path().join("fullmap.redb");
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        tablassert_rs::extract_prebuilt_fullmap(py, archive_path.clone(), output.clone(), None)
            .unwrap();
    });

    // 4. The primary landed at `output` and exactly s0..s15 exist beside it.
    assert!(output.exists(), "primary must land at the output path");
    for index in 0..common::SHARD_COUNT {
        let shard = common::shard_path(&output, index);
        assert!(shard.exists(), "missing extracted shard s{index}");
    }
    assert!(
        !common::shard_path(&output, common::SHARD_COUNT).exists(),
        "s16 must not exist"
    );

    // 5. The temp dir is removed on success: no `.fullmap.prebuilt-extract.d`
    //    (and no dot-prefixed stray at all) may remain in the output dir.
    assert!(
        !out_dir.path().join(".fullmap.prebuilt-extract.d").exists(),
        "temp dir must be removed after a successful extraction"
    );
    let strays: Vec<String> = std::fs::read_dir(out_dir.path())
        .unwrap()
        .map(|item| item.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with('.'))
        .collect();
    assert!(
        strays.is_empty(),
        "no dot-prefixed files may remain in the output dir, found: {strays:?}"
    );

    // 6. Equivalence pin: the extracted bundle must be canonically identical
    //    to the force-built fixture.
    let built = common::canonical_dump(&common::term_curie_map(&fixture_primary));
    let extracted = common::canonical_dump(&common::term_curie_map(&output));
    assert!(
        !built.is_empty(),
        "fixture must produce at least one indexed term"
    );
    assert_eq!(
        extracted, built,
        "extracted prebuilt archive must match the force build exactly"
    );
}
