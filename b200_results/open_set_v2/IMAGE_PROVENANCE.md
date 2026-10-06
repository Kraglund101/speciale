# Diffusion image provenance (seed 42)

Setting for all images: noise strength 0.4, 50 denoising steps, timesteps 381..1 evenly spaced (= noise-strength arm ns_040_s50).

- The 7,939 images listed in `images_made_with_ffe1fea_scheduler.txt` (generated 2026-10-04 09:30-13:01 on the B200) were
  made with commit ffe1fea: noise_strength_thesis_sets.fixed_scheduler + s35_timesteps(0.4) with 50 steps, i.e. exactly the
  code that produced the noise-strength test sets.
- All later images use pregenerate_synthetic.GEN (even_steps=True, EvenStepDDIM in src/inference/generate.py, commit df5c916),
  which per the Windows-side check reproduces ns_040_s50 within a max pixel difference of 2/255.
- Decision (server session, 2026-10-04 19:40): keep the 7,939 images (OPEN_SET_ACCEPT_EXISTING=1) instead of regenerating,
  so that one full seed fits into the 12-hour job. Regenerating them would cost ~4 h.
