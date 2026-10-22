# TODO

Running list of outstanding work on SideSolver, gathered while adding
`solve_puzzle_from_image` and retraining the digit classifier.

## Bugs / correctness

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
- [ ] The floodfilling approach used to find the grid (`flood_filling` /
      `find_biggest_blob`) is acknowledged in the README as a temp fix and breaks on
      low quality photos where corners can't be detected cleanly. README says this
      should be replaced with a Hough transform approach.
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
- [ ] `draw_solution_on_image` + `plt.show()` in `solve_puzzle_from_image` ignore the
      module's existing `display_images_flag` convention used everywhere else in
      `puzzle_extractor.py`, and nothing saves the result to disk — so there's no
      headless-safe output. Should gate the `plt.show()` behind the flag and/or
      `cv2.imwrite` the overlay image somewhere by default.

## ML / accuracy

- [x] Retrained the digit classifier (98.68% test accuracy, see `models/`).
- [ ] Only verified end-to-end accuracy on a single test image (`image1081`).
      Should run `extract_digits_from_image`/accuracy reporting across the full
      `v2_test_raw` set to get a real aggregate accuracy number now that the
      off-by-one bug is fixed.
- [ ] Decide what to do with the now-redundant model files in `models/`:
      `finalized_model_retrained.keras` (duplicate of the new `finalized_model.keras`)
      and `finalized_model_previous_backup.keras` (old model, kept as a rollback
      safety net). Delete once confident, or keep as a deliberate versioned backup.

## Dev environment

- [ ] `SudokuSolver/.venv` had a broken `python3` symlink pointing at a path that
      only existed on the machine it was created on (site-packages were intact,
      just the interpreter symlink was dead). Fixed for this session via
      `uv python install 3.11` + relinking, but this will bite again on a fresh
      clone/machine. Worth documenting proper venv setup (or re-pinning via
      `uv`/`requirements.txt`) so this is reproducible.
- [ ] `test/test_puzzle_extractor.py` exists but is empty (1 line) — essentially no
      automated test coverage for the extraction/classification/solving pipeline.

## Housekeeping

- [ ] Commit the pending changes: `solver.solve_puzzle`, the
      `solve_puzzle_from_image`/`draw_solution_on_image` additions, the
      `predict_number` off-by-one fix, and the README updates — all currently
      uncommitted on `upgrade_stack`.
