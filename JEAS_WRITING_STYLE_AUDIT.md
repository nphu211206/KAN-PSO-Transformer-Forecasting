# JEAS Writing Style Audit

Audit date: 2026-08-19.

## Evidence base

The audit uses archived publisher full text for four closely related JEAS
articles and the previously inspected KAN-NSGA-III article:

- MVMD-GNN-Transformer failure forecasting, DOI `10.1186/s44147-024-00543-2`.
- PSO-GBM pavement cracking, DOI `10.1186/s44147-025-00623-x`.
- PSO-SVR pavement roughness, DOI `10.1186/s44147-025-00706-9`.
- ACKLSTM-KAN wind forecasting, DOI `10.1186/s44147-025-00771-0`.
- KAN-NSGA-III machining optimization, DOI `10.1186/s44147-025-00729-2`.

The accepted articles are evidence of editorial tolerance, not prose templates.
Some contain awkward grammar, repetitive superiority language or claims broader
than their validation supports. Those weaknesses should not be copied.

## Observed JEAS prose pattern

1. The abstract normally follows problem -> method/data -> numerical result ->
   engineering implication in one self-contained block.
2. The introduction opens with an operational engineering need, reviews model
   families, identifies a concrete gap and ends with contributions or aims.
3. Methods use present tense for model definitions and past tense for executed
   data collection or experiments.
4. Results state the comparison and value together. Tables and figures are
   referenced directly rather than paraphrased for several paragraphs.
5. Discussion connects prediction error to maintenance, dispatch, reliability
   or process decisions and then states limitations.
6. Conclusions repeat only the main evidence chain and future scope; they do not
   introduce new metrics.

## House style selected for this manuscript

- Use American English consistently: `optimization`, `modeling`, `behavior`,
  `standardization` and `characterize`.
- Prefer a named subject and verb: `DLinear has the lowest mean...` rather than
  `It can clearly be observed that...`.
- Put the protocol, sample size and metric close to every central result.
- Use `shows`, `supports`, `is consistent with` and `does not establish` for
  evidence strength. Reserve `significant` for a defined statistical test.
- Use first person only for deliberate study actions: `We compare`, `we retain`
  and `we do not pool`. Avoid ceremonial phrases such as `In this esteemed
  study` or `It is worth noting that`.
- Vary sentence length through natural technical structure, not decorative
  synonyms. Keep one main claim per sentence where possible.
- Avoid advertising terms such as `groundbreaking`, `remarkable`, `superior`,
  `highly robust` or `state of the art` unless a registered comparison directly
  supports them.
- Preserve negative results and boundary statements. This is the manuscript's
  authentic voice and strongest distinction from formulaic winning-model papers.

## Changes applied

- Shortened the title from 20 to 16 words and made the technical subject direct.
- Reduced the abstract from 376 to approximately 250 words while retaining the
  dataset, protocol, main results, compute cost and claim boundary.
- Replaced ambiguous `preregistered` wording with `pre-specified` and `frozen
  before execution` because the protocols were frozen locally rather than
  deposited in a public preregistration service.
- Retained the AI-tool disclosure required by the archived JEAS guidance.
- Kept numerical claims conservative and avoided unsupported significance or
  universal-superiority language.

## AI-authorship and language risk

No prose style can guarantee the output of an AI detector, and attempting to
optimize for such detectors is not a scientific objective. The defensible
standard is authorship transparency: the author must understand, verify and
approve every sentence, while assistive-tool use is disclosed. The current
manuscript is evidence-specific and contains project-specific protocol,
failure-mode and provenance detail that generic generated prose normally lacks.

Before submission, the author should read the title, abstract, final paragraph
of the introduction, discussion and conclusion aloud and revise any sentence
that does not match his intended meaning. This human technical review is more
important than adding stylistic variation for its own sake.
