//! Integration tests for `extract_prebuilt_fullmap`.
//!
//! ## Positive round-trip
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
//! ## Negative / edge coverage (US-002)
//!
//! WHY: a prebuilt archive arrives over the network and may be torn, corrupt,
//! outdated, or outright hostile.  The extraction contract is that a failure
//! must (a) raise an error carrying ACTIONABLE context (what is wrong and how
//! to recover), (b) leave NO primary/shards beside `output` (validation
//! precedes every rename), and (c) leave NO `.<stem>.prebuilt-extract.d`
//! residue (cleanup runs on every error path).  Every negative test pins all
//! three, because a "failure" that still lands a broken DB or leaks multi-GB
//! partials is worse than no extraction at all.
//!
//! Fixtures and canonical-form helpers are shared with `build_golden` via
//! `tests/common/mod.rs` so both tests compare against one definition of the
//! canonical form.

mod common;

use std::fs::File;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a zstd-compressed tar at `archive_path`; `populate` appends the
/// members (plain `append_file` for fixture files, manual headers for hostile
/// entries).  Flushes the tar and finalizes the zstd frame so the archive is
/// complete on disk.
fn write_tar_zst_with(
    archive_path: &Path,
    populate: impl FnOnce(&mut tar::Builder<zstd::Encoder<'static, File>>),
) {
    let archive_file = File::create(archive_path).unwrap();
    let encoder = zstd::Encoder::new(archive_file, 0).unwrap();
    let mut builder = tar::Builder::new(encoder);
    populate(&mut builder);
    let encoder = builder.into_inner().unwrap();
    encoder.finish().unwrap();
}

/// Package `(member name, source file)` pairs into a zstd-compressed tar at
/// `archive_path` (flat layout, no directory entries).
fn package_tar_zst(archive_path: &Path, members: &[(String, PathBuf)]) {
    write_tar_zst_with(archive_path, |builder| {
        for (name, source) in members {
            let mut file = File::open(source).unwrap();
            builder.append_file(name, &mut file).unwrap();
        }
    });
}

/// `(member name, file path)` pairs for shards `0..count` of
/// `fixture_primary` (member names identical to the on-disk file names).
fn shard_members(fixture_primary: &Path, count: usize) -> Vec<(String, PathBuf)> {
    (0..count)
        .map(|index| {
            let shard = common::shard_path(fixture_primary, index);
            (
                shard.file_name().unwrap().to_string_lossy().into_owned(),
                shard,
            )
        })
        .collect()
}

/// Run `extract_prebuilt_fullmap` exactly as Python callers do and require it
/// to fail; assert the error message carries every expected context fragment
/// (each negative must tell the user WHAT is wrong and HOW to recover).
fn extract_expect_error(archive: &Path, output: &Path, fragments: &[&str]) {
    pyo3::Python::initialize();
    let error = pyo3::Python::attach(|py| {
        tablassert_rs::extract_prebuilt_fullmap(
            py,
            archive.to_path_buf(),
            output.to_path_buf(),
            None,
        )
        .expect_err("extraction of a broken/hostile archive must fail")
        .to_string()
    });
    for fragment in fragments {
        assert!(
            error.contains(fragment),
            "error {error:?} must carry the actionable context {fragment:?}"
        );
    }
}

/// Assert a failed extraction landed NOTHING: no primary at `output`, no
/// sibling shards, and no dot-prefixed residue (the
/// `.<stem>.prebuilt-extract.d` temp dir included) in the output dir — a
/// failed extraction must never leave multi-GB partials behind.
fn assert_failed_extraction_left_nothing(out_dir: &Path, output: &Path) {
    assert!(
        !output.exists(),
        "no primary may land at {} after a failed extraction",
        output.display()
    );
    for index in 0..common::SHARD_COUNT {
        let shard = common::shard_path(output, index);
        assert!(
            !shard.exists(),
            "no shard s{index} may land beside the output after a failed extraction"
        );
    }
    if out_dir.exists() {
        let strays: Vec<String> = std::fs::read_dir(out_dir)
            .unwrap()
            .map(|item| item.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.starts_with('.'))
            .collect();
        assert!(
            strays.is_empty(),
            "a failed extraction must leave no dot-prefixed residue in {}, found: {strays:?}",
            out_dir.display()
        );
    }
}

// ---------------------------------------------------------------------------
// Positive round-trip (US-001)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Negatives: corrupt / torn archives
// ---------------------------------------------------------------------------

/// WHY: a torn download (garbage bytes, not even a zstd frame) must fail at
/// the zstd layer BEFORE any extraction work — and, like every failure, leave
/// no primary, no shards, and no temp-dir residue beside `output`.
#[test]
fn corrupt_archive_is_rejected_and_leaves_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("fullmap.tar.zst");
    std::fs::write(&archive, b"not zstd at all").unwrap();
    let out_dir = tempfile::tempdir().unwrap();
    let output = out_dir.path().join("fullmap.redb");
    extract_expect_error(&archive, &output, &["not valid zstd"]);
    assert_failed_extraction_left_nothing(out_dir.path(), &output);
}

/// WHY: an archive holding ONLY shards (the primary lost in a torn upload)
/// cannot serve a single lookup; the absence must be named explicitly instead
/// of failing later with a cryptic open error.
#[test]
fn archive_without_primary_is_rejected() {
    let fixture_dir = tempfile::tempdir().unwrap();
    let fixture_primary = common::build_fixture(fixture_dir.path(), 1);
    let archive = fixture_dir.path().join("fullmap.tar.zst");
    package_tar_zst(
        &archive,
        &shard_members(&fixture_primary, common::SHARD_COUNT),
    );
    let out_dir = tempfile::tempdir().unwrap();
    let output = out_dir.path().join("fullmap.redb");
    extract_expect_error(&archive, &output, &["no primary"]);
    assert_failed_extraction_left_nothing(out_dir.path(), &output);
}

/// WHY: losing shard s15 mid-transfer is the classic torn-archive shape; the
/// error must name the MISSING shard so operators know what to re-fetch
/// (silently landing 15 shards would make every s15 term unfindable).
#[test]
fn missing_shard_is_rejected_and_named() {
    let fixture_dir = tempfile::tempdir().unwrap();
    let fixture_primary = common::build_fixture(fixture_dir.path(), 1);
    let mut members = vec![("fullmap.redb".to_string(), fixture_primary.clone())];
    members.extend(shard_members(&fixture_primary, common::SHARD_COUNT - 1)); // s0..s14
    let archive = fixture_dir.path().join("fullmap.tar.zst");
    package_tar_zst(&archive, &members);
    let out_dir = tempfile::tempdir().unwrap();
    let output = out_dir.path().join("fullmap.redb");
    extract_expect_error(
        &archive,
        &output,
        &["inconsistent shard set", "missing [15]"],
    );
    assert_failed_extraction_left_nothing(out_dir.path(), &output);
}

/// WHY: an s16 beside a 16-shard primary means wrong or torn packaging; the
/// unexpected shard must be named (its contents are never even read — the
/// shard-set check fires first).
#[test]
fn extra_shard_is_rejected_and_named() {
    let fixture_dir = tempfile::tempdir().unwrap();
    let fixture_primary = common::build_fixture(fixture_dir.path(), 1);
    // No s16 exists in a real build; misuse a copy of s0's bytes as the stray.
    let stray = fixture_dir.path().join("fullmap.s16.redb");
    std::fs::copy(common::shard_path(&fixture_primary, 0), &stray).unwrap();
    let mut members = vec![("fullmap.redb".to_string(), fixture_primary.clone())];
    members.extend(shard_members(&fixture_primary, common::SHARD_COUNT));
    members.push(("fullmap.s16.redb".to_string(), stray));
    let archive = fixture_dir.path().join("fullmap.tar.zst");
    package_tar_zst(&archive, &members);
    let out_dir = tempfile::tempdir().unwrap();
    let output = out_dir.path().join("fullmap.redb");
    extract_expect_error(
        &archive,
        &output,
        &["inconsistent shard set", "unexpected [16]"],
    );
    assert_failed_extraction_left_nothing(out_dir.path(), &output);
}

/// WHY: a nonexistent archive path (wrong flag, unfetched file) must produce
/// a plain "not found" error BEFORE any output-dir side effects, so a retry
/// with the right path starts clean.
#[test]
fn missing_archive_is_a_not_found_error() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("does-not-exist.tar.zst");
    let out_dir = tempfile::tempdir().unwrap();
    let output = out_dir.path().join("fullmap.redb");
    extract_expect_error(&archive, &output, &["not found"]);
    assert_failed_extraction_left_nothing(out_dir.path(), &output);
}

// ---------------------------------------------------------------------------
// Negatives: invalid database content
// ---------------------------------------------------------------------------

/// Downgrade a COPY of the primary to the v1 schema tag, exactly the shape an
/// outdated published archive would carry.  `Database::open` takes a WRITABLE
/// handle on the copy (the `open_primary_copy` precedent, minus the read-only
/// use): overwrite META.schema, commit, drop.  The original stays untouched
/// (and flock-locked in the DB cache), hence the copy.
fn outdated_primary_copy(fixture_primary: &Path) -> PathBuf {
    let outdated = fixture_primary.with_file_name("outdated.redb");
    std::fs::copy(fixture_primary, &outdated).unwrap();
    let db = redb::Database::open(&outdated).unwrap();
    let txn = db.begin_write().unwrap();
    {
        let mut meta = txn.open_table(common::META).unwrap();
        meta.insert("schema", "tablassert.fullmap.v1").unwrap();
    }
    txn.commit().unwrap();
    drop(db);
    outdated
}

/// WHY: prebuilt archives outlive schema bumps; extracting a v1..v4 bundle
/// would land a DB that every lookup immediately rejects.  The error must
/// carry the wrapped `validate_schema` demand to rebuild, and nothing may
/// land.
#[test]
fn outdated_schema_is_rejected_and_demands_rebuild() {
    let fixture_dir = tempfile::tempdir().unwrap();
    let fixture_primary = common::build_fixture(fixture_dir.path(), 1);
    let outdated = outdated_primary_copy(&fixture_primary);
    let mut members = vec![("fullmap.redb".to_string(), outdated)];
    members.extend(shard_members(&fixture_primary, common::SHARD_COUNT));
    let archive = fixture_dir.path().join("fullmap.tar.zst");
    package_tar_zst(&archive, &members);
    let out_dir = tempfile::tempdir().unwrap();
    let output = out_dir.path().join("fullmap.redb");
    extract_expect_error(
        &archive,
        &output,
        &["failed validation", "outdated", "build-fullmap"],
    );
    assert_failed_extraction_left_nothing(out_dir.path(), &output);
}

/// WHY: a byte-corrupted (or swapped) primary must fail at `open_read_only`
/// with the validation context — never land — even when all 16 shards are
/// intact and valid.
#[test]
fn non_redb_primary_is_rejected() {
    let fixture_dir = tempfile::tempdir().unwrap();
    let fixture_primary = common::build_fixture(fixture_dir.path(), 1);
    let garbage = fixture_dir.path().join("garbage.redb");
    std::fs::write(&garbage, b"this is definitely not a redb database").unwrap();
    let mut members = vec![("fullmap.redb".to_string(), garbage)];
    members.extend(shard_members(&fixture_primary, common::SHARD_COUNT));
    let archive = fixture_dir.path().join("fullmap.tar.zst");
    package_tar_zst(&archive, &members);
    let out_dir = tempfile::tempdir().unwrap();
    let output = out_dir.path().join("fullmap.redb");
    extract_expect_error(&archive, &output, &["failed validation"]);
    assert_failed_extraction_left_nothing(out_dir.path(), &output);
}

// ---------------------------------------------------------------------------
// Negatives: hostile archive entries
// ---------------------------------------------------------------------------

/// WHY: a hostile `../evil` member must be rejected by the explicit path
/// validation — tar-rs's own `unpack_in` protection SILENTLY SKIPS escaping
/// members, but this contract is a LOUD error — and nothing may be written
/// outside the temp dir: neither beside `output` nor in its parent.
#[test]
fn path_traversal_entry_is_rejected_and_writes_nothing_outside() {
    let out_root = tempfile::tempdir().unwrap();
    let archive = out_root.path().join("fullmap.tar.zst");
    write_tar_zst_with(&archive, |builder| {
        // tar-rs's `set_path` refuses to WRITE `..` components (write-side
        // safety), so smuggle them in by patching the raw name field after a
        // benign path — exactly the bytes a hostile archiver would ship.
        let mut header = tar::Header::new_gnu();
        header.set_path("evil").unwrap();
        header.set_entry_type(tar::EntryType::Regular);
        header.set_size(4);
        header.set_mode(0o644);
        header.as_mut_bytes()[..7].copy_from_slice(b"../evil");
        header.set_cksum();
        builder.append(&header, &b"evil"[..]).unwrap();
    });
    // Output nested one level deep so BOTH potential escape targets — beside
    // `output` (db/evil) and beside its parent (evil) — stay observable.
    let out_dir = out_root.path().join("db");
    let output = out_dir.join("fullmap.redb");
    extract_expect_error(&archive, &output, &["unsafe path", "../evil"]);
    assert_failed_extraction_left_nothing(&out_dir, &output);
    assert!(
        !out_dir.join("evil").exists(),
        "traversal must not write beside the output"
    );
    assert!(
        !out_root.path().join("evil").exists(),
        "traversal must not escape into the output dir's parent"
    );
}

/// WHY: a symlink member could point extraction outside the temp dir or fake
/// a `.redb` without real bytes; the contract admits only directories and
/// regular files, so any other entry type errors loudly before any scan.
#[test]
fn symlink_entry_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("fullmap.tar.zst");
    write_tar_zst_with(&archive, |builder| {
        let mut header = tar::Header::new_gnu();
        header.set_path("evil-link.redb").unwrap();
        header.set_entry_type(tar::EntryType::Symlink);
        header.set_link_name("fullmap.redb").unwrap();
        header.set_size(0);
        header.set_cksum();
        builder.append(&header, std::io::empty()).unwrap();
    });
    let out_dir = tempfile::tempdir().unwrap();
    let output = out_dir.path().join("fullmap.redb");
    extract_expect_error(&archive, &output, &["unsupported type", "Symlink"]);
    assert_failed_extraction_left_nothing(out_dir.path(), &output);
}

/// WHY: tar-rs IGNORES `GNU.sparse.*` PAX records (bsdtar's sparse encoding),
/// and redb files are genuinely sparse (preallocated), so a bsdtar-created
/// PAX archive would silently extract WRONG bytes.  The extractor therefore
/// rejects any archive carrying those records; this test builds one the way
/// bsdtar would — a PAX extended header (typeflag 'x') whose records describe
/// the NEXT entry.
#[test]
fn pax_sparse_archive_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("fullmap.tar.zst");
    write_tar_zst_with(&archive, |builder| {
        // PAX record format: "<len> <key>=<value>\n" where <len> counts the
        // whole record including the length itself:
        // "24 GNU.sparse.size=4096\n" is exactly 24 bytes.
        let records = b"24 GNU.sparse.size=4096\n";
        let mut pax_header = tar::Header::new_gnu();
        pax_header.set_path("PaxHeader/fullmap.redb").unwrap();
        pax_header.set_entry_type(tar::EntryType::XHeader);
        pax_header.set_size(records.len() as u64);
        pax_header.set_mode(0o644);
        pax_header.set_cksum();
        builder.append(&pax_header, &records[..]).unwrap();

        // The regular entry the PAX header describes.
        let mut file_header = tar::Header::new_gnu();
        file_header.set_path("fullmap.redb").unwrap();
        file_header.set_entry_type(tar::EntryType::Regular);
        file_header.set_size(4);
        file_header.set_mode(0o644);
        file_header.set_cksum();
        builder.append(&file_header, &b"data"[..]).unwrap();
    });
    let out_dir = tempfile::tempdir().unwrap();
    let output = out_dir.path().join("fullmap.redb");
    extract_expect_error(
        &archive,
        &output,
        &["PAX sparse records", "GNU.sparse.size"],
    );
    assert_failed_extraction_left_nothing(out_dir.path(), &output);
}

// ---------------------------------------------------------------------------
// Negatives + prefer-pin: multiple primaries
// ---------------------------------------------------------------------------

/// WHY: two non-shard .redb members (neither named exactly `fullmap.redb`)
/// mean torn or mispackaged bytes; before the US-002 hardening the unchosen
/// candidate was silently DISCARDED.  Extraction now refuses and lists every
/// candidate by name.
#[test]
fn multiple_unnamed_primaries_are_rejected_and_listed() {
    let fixture_dir = tempfile::tempdir().unwrap();
    let fixture_primary = common::build_fixture(fixture_dir.path(), 1);
    let stray_a = fixture_dir.path().join("primary_a.redb");
    let stray_b = fixture_dir.path().join("primary_b.redb");
    std::fs::copy(&fixture_primary, &stray_a).unwrap();
    std::fs::copy(&fixture_primary, &stray_b).unwrap();
    let mut members = vec![
        ("primary_a.redb".to_string(), stray_a),
        ("primary_b.redb".to_string(), stray_b),
    ];
    members.extend(shard_members(&fixture_primary, common::SHARD_COUNT));
    let archive = fixture_dir.path().join("fullmap.tar.zst");
    package_tar_zst(&archive, &members);
    let out_dir = tempfile::tempdir().unwrap();
    let output = out_dir.path().join("fullmap.redb");
    extract_expect_error(
        &archive,
        &output,
        &[
            "2 primary .redb candidates",
            "primary_a.redb",
            "primary_b.redb",
        ],
    );
    assert_failed_extraction_left_nothing(out_dir.path(), &output);
}

/// WHY: the documented prefer-case must SURVIVE the multi-primary hardening —
/// when exactly one candidate is literally named `fullmap.redb`, it wins over
/// strays (the historical selection rule).  Extraction succeeds, the landed
/// bundle matches the force build, and the stray is discarded with the temp
/// dir.
#[test]
fn named_fullmap_primary_is_preferred_over_strays() {
    let fixture_dir = tempfile::tempdir().unwrap();
    let fixture_primary = common::build_fixture(fixture_dir.path(), 1);
    let stray = fixture_dir.path().join("stray.redb");
    std::fs::copy(&fixture_primary, &stray).unwrap();
    let mut members = vec![
        ("fullmap.redb".to_string(), fixture_primary.clone()),
        ("stray.redb".to_string(), stray),
    ];
    members.extend(shard_members(&fixture_primary, common::SHARD_COUNT));
    let archive = fixture_dir.path().join("fullmap.tar.zst");
    package_tar_zst(&archive, &members);
    let out_dir = tempfile::tempdir().unwrap();
    let output = out_dir.path().join("fullmap.redb");
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        tablassert_rs::extract_prebuilt_fullmap(py, archive.clone(), output.clone(), None).unwrap();
    });
    assert!(
        !out_dir.path().join("stray.redb").exists(),
        "the stray primary must be discarded with the temp dir, not landed"
    );
    let built = common::canonical_dump(&common::term_curie_map(&fixture_primary));
    let extracted = common::canonical_dump(&common::term_curie_map(&output));
    assert_eq!(
        extracted, built,
        "the preferred `fullmap.redb` primary must land intact"
    );
}
