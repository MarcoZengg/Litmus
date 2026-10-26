# Labeling guide

Consistent labels are what make Litmus scores mean something. Read this once
before labeling, and keep it open while you work. Field definitions are in
[design.md, section 5](design.md#5-dataset-format). Examples here use the
invented X200 air purifier.

## The golden rule

**Label what the ideal agent should do, given its knowledge base. Never label
what the current agent does.** If labels copy the agent's behavior, you're
grading the agent against itself, and every score is meaningless.

## For each question

### 1. `source`: where did the question come from?

| Value | Use when |
|---|---|
| `real_traffic` | A real customer actually asked it (chat log, email, phone call). Still `real_traffic` if someone tidied the wording; say so in `notes` |
| `staff` | Someone on the team wrote it, e.g. questions they expect customers to ask |
| `faq` | Copied from a published FAQ, and you don't know who first asked it |
| `synthetic` | Written or rewritten by an LLM |
| `mixed` | You don't know. Better than guessing |

### 2. `expected_behavior`: what should the ideal agent do?

Work through these in order and stop at the first that applies:

1. **Is it about something the agent covers?** If not, `refuse`.
   *"What's the weather tomorrow?"*
2. **Is it clear enough to answer?** If the product or model is missing and it
   matters, `clarify`. *"What's the battery life?"*, when there are several
   products.
3. **Does the knowledge base answer it?**
   - Every part: `answer`.
   - Some parts: `partial`.
   - No part, but a person could help (prices, orders, one site's setup):
     `handoff`. *"Does the X200 at my site have the HEPA upgrade?"*

If the reply should contain any real answer, label it `answer` or `partial`,
even if it also suggests contacting a person.

**Label with the run setup in mind.** If the request also sends
`metadata`, such as a product category, the app knows the product, so the
question isn't vague.

### 3. `expected_output`: the reference answer

- Write the facts a correct answer must contain, briefly. It isn't the only
  acceptable wording.
- Keep numbers, units, ranges and conditions exactly ("optional", "usually",
  "4–6 hours").
- Leave it empty for `clarify`, `refuse` and `handoff`.

### 4. `reference_context_ids`: the correct passages

- Use the passage IDs the app itself returns.
- `["p12"]` means p12 is needed. `[["p4", "faq-7"], "p15"]` means it needs
  (p4 or faq-7) and p15.
- Use `[]` when no passage should answer: `clarify`, `refuse`, `handoff`.
- Leave the field out if you don't know. That's better than a guess.

### 5. `notes`

Write down anything you were unsure about, and every judgment call, so the
next labeler decides the same way.

## Tricky cases

| Case | Label |
|---|---|
| Multi-part question, e.g. "How loud is it, and how often do I change the filter?" | `answer` only if the knowledge base covers **every** part; otherwise `partial` |
| Optional feature: "Can the X200 have a HEPA upgrade?" | `answer`: the answer is "yes, it's optional" |
| Installed feature: "Does **my** X200 have the HEPA upgrade?" | `handoff`: the documents can't know one site's setup |
| Company or marketing question: "Why choose you over competitors?" | Decide once per project, write the decision in `notes`, and apply it every time |
| Typo in a source answer | Fix the obvious typo, and say so in `notes` |

## Habits that keep labels consistent

- **Label one category at a time,** so similar questions get similar labels.
- **Re-read your first 10 after finishing them.** Early labels drift the most.
- **When unsure,** write the doubt in `notes` and keep going. Settle all the
  doubtful ones together at the end.
- **Time yourself,** to estimate how long the next batch will take.
- **Don't change a label after seeing the agent's results,** unless the label
  was actually wrong. If you change one, write why in `notes`. Otherwise you
  end up tuning the test to the agent.

## Labeling replies to check the judge

This is a second kind of labeling. The sections above label the dataset's
questions; this one grades the app's actual replies, so Litmus can measure how
often the judge agrees with you (design doc, section 7).

1. Run `litmus label <run> --metric correctness`. It writes
   `labels-correctness.csv` in the run folder with up to 50 replies.
2. Open it in a spreadsheet app. For each row, read the question, the reference
   answer and the app's answer, and write `pass` or `fail` in `label`. Use
   `note` for doubts. Leave a row blank to skip it.
3. Save it as CSV, in the same place.
4. Judge the run if you haven't yet. `litmus report` shows the agreement.

Grade by the judge's rules, not by taste:

- **Correctness `pass`:** the answer gives the facts the question asks about,
  agrees with the reference and contradicts nothing in it. Wording, length,
  language, politeness and extra details don't matter, unless an extra detail
  contradicts the reference.
- **Correctness `fail`:** a fact the question asks about is missing or wrong,
  the answer contradicts the reference, or it doesn't address the question.
- **Faithfulness:** `pass` only if every factual claim in the answer is
  supported by the passages in the sheet. Don't use what you know yourself.
- **Behavior:** pick one of the five behaviors, using the definitions above.

Habits:

- **Label before you look at the judge's verdicts** (`judgments.jsonl` or the
  report's "Incorrect answers"). Otherwise you grade the judge, not the answer.
- **Don't change reference answers to make the judge look better.** If one is
  actually wrong, fix it, bump the dataset version, and label a new run.
- **A cell starting with `'`** was protected from being read as a formula. The
  `'` isn't part of the answer.
