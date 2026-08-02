# P1 Reflection — nanoGPT

Week 4

In this week, I worked on some of the pending work on building nanoGPT from scratch, a P1 that has been part of Week 1. The file `train.py` had been sitting at 0 bytes the whole time — P1 was never blocked on compute or on understanding, it was blocked on nobody opening the file. That is the part worth remembering.

I encountered some challenges. The time estimate for completion was in the range of 15–25 minutes and it ended up being six hours. The cause printed on the first line of the run: `device: cpu`. The venv had a CPU-only torch build (`2.12.0+cpu`), so the RTX 3060 was never reachable. Neither of us stopped to check it before committing to the run.

The training itself worked. Loss went from 4.24 to 1.72 train / 1.87 val over 1000 iterations, with the train/val gap only opening late — mild overfitting starting, nothing serious. At 1.87 the model is undertrained; reaching ~1.5 would need roughly 5x the iterations, which is a GPU job, not a CPU one.

The samples are where the real lesson is. The model produced `JUIEDA:` — an invented speaker name, correctly capitalised, colon-terminated, followed by an indented verse block. Line lengths are right. Commas fall in plausible places. `'ell` appears as an elision. But most of the words do not exist: *suchagnled*, *arentilong*, *caunterviyng* — phonotactically valid English that was never written. Only the highest-frequency fragments are real: *I pray*, *and when*, *of youth*.

Form is learned before meaning. Structure is the most predictable signal in the data, so next-token prediction picks it up first and cheaply; vocabulary and sense cost far more capacity and far more steps. A loss of 1.87 versus 1.5 sounds like a small numeric difference and is actually the difference between texture and language.

What I would do differently: check `torch.cuda.is_available()` before starting any run, and open every file a task claims to depend on before assuming the task is blocked on something harder.
