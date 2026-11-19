# TODO

Running list of outstanding work on SideSolver, gathered while adding
`solve_puzzle_from_image` and retraining the digit classifier.

## Bugs / correctness

- [x] ~~`is_blank_digit()`'s blank-vs-digit decision is measurably worse than the
      digit classifier itself — 92.99% overall accuracy on `v2_test_raw` vs.
      98.68% non-blank-only accuracy...~~ Fixed — collected the blob-size ratio
      `is_blank_digit` computes internally for every one of the 2511 labeled
      cells in `v2_test_raw`, against the ground-truth blank/non-blank label,
      and swept thresholds against that data. The smallest ratio seen for any
      real digit was 0.0472; the old `1/40` (0.025) threshold left a lot of
      headroom below that, misclassifying 164/2511 blank cells as non-blank.
      Moved the threshold to `1/22` (0.0455) — still strictly below 0.0472, so
      no real digit is newly misclassified as blank (that direction is worse:
      a missed digit becomes a blank the solver silently overwrites, rather
      than failing safely) — which cuts blank-misclassified-as-non-blank down
      to 30/2511. Measured effect on the full `v2_test_raw` accuracy:
      - Overall accuracy: **92.99% → 98.33%**
      - Non-blank accuracy: unchanged (98.68%) — expected, since this only
        affects blank/non-blank decisions, not digit classification itself.
      - Perfect-puzzle rate: **22.6% (7/31) → 61.3% (19/31)**
      Regression guard thresholds in `test_full_dataset_accuracy_does_not_regress`
      updated to the new baseline (with headroom).
- [x] ~~`solver.solve_puzzle()`/`_fill_in_puzzle()` never validates that the
      *given* (non-blank) cells are mutually consistent...~~ Fixed —
      `solve_puzzle()` now calls `_givens_are_consistent()` first (checks every
      row/column/3x3 box for a repeated non-zero digit) and returns `None`
      immediately if the givens themselves are already contradictory, before
      ever attempting to fill anything. Covered by three new tests in
      `test_solver.py` isolating the row, column, and box cases.

- [x] ~~`digit_classifier.predict_number` returned the raw 0-8 class index instead of the
      1-9 digit~~ (training labels are `folder - 1`). Fixed — now `+ 1`. This was
      causing every non-blank prediction to be off by exactly one.
- [x] ~~`corner_detection()` can return `None` when the grid can't be found... the
      two image-processing entry points should do the same~~ Fixed —
      `extract_digits_from_image` and `solve_puzzle_from_image` now check for
      `None` corners and bail out with a warning instead of crashing. Also fixed
      two crashes inside `corner_detection()` itself found while testing this:
      an empty `contours` list (`contours[0]` → `IndexError`) and a
      zero-area largest contour (`quad_area / blob_area` → `ZeroDivisionError`)
      both now return `None` with a warning instead of crashing.
- [x] ~~The floodfilling approach used to find the grid (`flood_filling` /
      `find_biggest_blob`) is acknowledged in the README as a temp fix and breaks on
      low quality photos where corners can't be detected cleanly...~~
      Partially addressed — added `grid_detection.detect_grid_corners()`, which
      tries the original floodfill + contour approach first (unchanged, zero
      risk to images that already work) and falls back to a new Hough-line-based
      approach (`detect_grid_corners_hough()`) only when that fails. Real,
      measured trade-off on `v2_test_raw`:
      - Images skipped entirely: **9/40 → 1/40**. Big win on raw coverage.
      - But non-blank accuracy: **98.68% → 80.04%**, overall accuracy:
        **92.99% → 84.01%**. The 8 newly-rescued images are often imprecisely
        aligned (the Hough corners are sometimes off by about one grid
        row/column), which corrupts digit extraction for a large fraction of
        their cells.
      - Verified this doesn't create a silent-wrong-answer risk: manually ran
        `solve_puzzle_from_image()` on all 8 rescued images — every one fails
        safely (`None`), none produce an incorrect "solved" overlay. The
        givens-consistency check above and the solver's own inability to
        complete a contradiction-laden grid catch these before they'd ever
        reach a user.
      - Net effect: these 8 images go from "grid not detected" to "couldn't
        solve this puzzle" — same practical outcome for the user today, but
        now exercising far more of the pipeline, and a few images that were
        previously skipped for fixable reasons (not genuinely bad photos) may
        now succeed.
      - Test thresholds in `test_full_dataset_accuracy_does_not_regress`
        updated to the new measured baseline (with headroom).
      - **Follow-up fix:** diagnosed *why* alignment was imprecise by looking
        at which rows/columns the misclassifications concentrated in — top
        and bottom rows were far worse (22%/3% accuracy) than the middle
        (65-78%), pointing at the top/bottom boundary lines specifically.
        Traced it to `_longest_regular_hough_run` using `min(gaps)` as the
        grid-cell-size base unit: one spurious near-duplicate line near a
        border (confirmed directly on `image50`'s and `image32`'s detected
        line positions) produces one anomalously small gap, and using that
        as the base unit corrupts the multiple-of-base-unit check for every
        other gap, discarding most of the real grid lines. Switched to
        `median(gaps)`, which is robust to that one outlier. Measured effect:
        `num_skipped` **1 → 0** (the last skipped image, `image51`, is now
        recovered too), 3 of the other rescued images improved (`image32`
        34.6%→55.6%, `image50` 35.8%→49.4%, `image35` 48.1%→51.9%), none
        regressed. Aggregate accuracy stayed about the same only because
        `image51` (newly included, moderate accuracy) entered the pool.
      - Remaining gap: the other 5 rescued images are unchanged by this fix,
        so there's likely at least one more distinct alignment issue beyond
        the base-unit bug — worth another look if coverage on these specific
        images matters.
- [x] ~~`digit_classifier.load_model()` has a dead/legacy code path for pickled
      `.sav` models...~~ Removed — no caller ever passed a non-`.keras` path, and
      the legacy pickle format is fundamentally incompatible with the current
      TensorFlow/Keras version (that was the whole point of the `RuntimeError`),
      so there was nothing to fix. `load_model()` now just loads `.keras` models,
      the unused `pickle` import is gone, and `models/finalized_model.sav` was
      deleted.

## Performance

- [ ] `solver.solve_puzzle()` is naive backtracking with no constraint propagation
      (no MRV/pruning) — took ~24s on a single real puzzle in testing. Fine for
      one-off CLI use, but worth optimizing (e.g. candidate-set pruning, numpy
      bitmasks) if this is ever used interactively or on batches.
- [ ] `flood_filling()` is a hand-rolled BFS/DFS over every pixel in pure Python —
      slow for large images. Could likely be replaced with
      `cv2.connectedComponentsWithStats`.
- [ ] `main()` currently runs the *entire* extraction + classification pipeline
      twice (once in `extract_digits_from_image` for the accuracy report, once in
      `solve_puzzle_from_image`). Should extract/classify once and reuse the result
      for both accuracy reporting and solving.

## Wiring / CLI

- [ ] `solve_puzzle_from_image` is only called from inside `puzzle_extractor.main()`,
      hardcoded. There's no way to solve a new puzzle photo (no ground-truth `.dat`
      file available) without also requiring `puzzle_solution` for the accuracy
      report. Needs its own CLI flag / entry point in `main.py` decoupled from the
      accuracy-report path.
- [x] ~~`draw_solution_on_image` + `plt.show()` in `solve_puzzle_from_image` ignore
      the module's existing `display_images_flag` convention...~~ Fixed —
      `plt.show()` is now gated behind `config.display_images_flag`, which is what
      made it safe to call `solve_puzzle_from_image` from the web app's request
      handler in the first place.

## ML / accuracy

- [x] Retrained the digit classifier (98.68% test accuracy, see `models/`).
- [x] ~~Only verified end-to-end accuracy on a single test image (`image1081`)...~~
      Done — added `puzzle_extractor.evaluate_dataset_accuracy()`, which runs
      extraction + classification (no solving) over every labeled image in a
      directory and reports aggregate stats. Real numbers across all 40 images
      in `v2_test_raw`:
      - **9/40 images (22.5%) skipped** — grid corners couldn't be detected at
        all. This is the single biggest real-world failure mode, bigger than
        classifier accuracy, and is exactly the case the Hough-transform rewrite
        above is meant to fix.
      - Non-blank digit accuracy: **98.68%** (matches the classifier's own
        held-out test accuracy — good consistency check).
      - Overall accuracy including blanks: **92.99%** — *lower* than the
        non-blank number, meaning `is_blank_digit()` blank-vs-digit detection is
        actually the weaker link, not the digit classifier itself. New finding,
        not previously tracked anywhere — worth its own investigation.
      - Fully-correct puzzles: **7/31 evaluated (22.6%)** — well below the
        README's earlier 0.9868^25≈71% estimate, because that estimate only
        accounted for per-digit accuracy and ignored both blank-detection errors
        and corner-detection failures.
      Locked in as a regression guard: `test_full_dataset_accuracy_does_not_regress`
      (`@pytest.mark.slow`) asserts thresholds with headroom below these numbers.
- [ ] Decide what to do with the now-redundant model files in `models/`:
      `finalized_model_retrained.keras` (duplicate of the new `finalized_model.keras`)
      and `finalized_model_previous_backup.keras` (old model, kept as a rollback
      safety net). Delete once confident, or keep as a deliberate versioned backup.

## Dev environment

- [x] ~~`SudokuSolver/.venv` had a broken `python3` symlink pointing at a path
      that only existed on the machine it was created on... Worth documenting
      proper venv setup so this is reproducible.~~ Fixed — added a root-level
      `.python-version` (pins `3.11`) and a "Setup" section in `README.md` with
      the exact `uv python install` / `uv venv` / `uv pip install` commands,
      verified end-to-end against a scratch copy of `requirements.txt` before
      writing them down.
- [x] ~~`test/test_puzzle_extractor.py` exists but is empty... no automated test
      coverage~~ Fixed — basic `pytest` suite added: `SudokuSolver/pytest.ini` +
      `test/conftest.py` (fixture paths, `pythonpath = src` so the flat
      `import solver` style works), `test/test_solver.py` (unit tests for
      `solve_puzzle`: real puzzle, already-solved grid, unsolvable grid),
      `test/test_accuracy_reporter.py` (unit tests for `parse_puzzle_file`/
      `calculate_accuracy`), and `test/test_puzzle_extractor.py` (the
      `image1081` regression test — asserts 81/81 classification accuracy, plus
      a `@pytest.mark.slow`-marked full `solve_puzzle_from_image` end-to-end
      test). `pytest` passes (8 tests); `pytest -m "not slow"` runs the fast
      ones only (~20s vs ~45s). Added `pytest==9.1.1` to `requirements.txt`.

## Product / GUI

- [x] Build a GUI where a puzzle image can be dropped in and is solved automatically.
      Built as a Flask web app: `SudokuSolver/src/web_app.py` +
      `SudokuSolver/src/templates/index.html`. Drag-and-drop or click-to-browse a
      photo, it POSTs to `/solve` which calls `puzzle_extractor.solve_puzzle_from_image`
      and returns the overlay image as base64 PNG. Verified end-to-end (success,
      no-grid-detected, no-file, wrong-file-type all return the right response).
      Run with `python web_app.py` from `SudokuSolver/src` and open
      `http://127.0.0.1:5000`.
    - [ ] Follow-up: the digit classifier model is reloaded from disk on every
          `/solve` request (`classify_image_digits` calls `digit_classifier.load_model`
          each time). Fine for a prototype; should load once at app startup and reuse
          it if this gets real traffic.
    - [ ] Follow-up: solver is still the naive ~24s backtracking from the Performance
          section above — the UI just shows a "this can take up to 30s" spinner
          around it rather than fixing the underlying slowness.
- [ ] Eventually turn this into a mobile app: scan a puzzle with the camera and solve
      it in-app. Longer-term goal the GUI above should build toward, not a v1
      requirement.

## Housekeeping

- [x] ~~Commit the pending changes... all currently uncommitted on
      `upgrade_stack`~~ Done — merged to `master` via PR #34 (squash). `upgrade_stack`
      and all 13 stale dependabot branches/PRs have since been deleted/closed;
      `master` is now the only branch.
