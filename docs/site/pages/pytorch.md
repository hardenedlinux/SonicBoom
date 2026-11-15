# Coming from PyTorch

SonicBoom is not PyTorch, but many of its names and conventions carry over.
Type a PyTorch name below to see what the Guile binding offers for it:

- **supported** — there is a function with the same meaning; the note says how
  it differs, if it does.
- **recipe** — there is no single function, but the operation is a few lines of
  existing ones; the recipe runs as written.
- **missing** — nothing in the binding does this yet.
- **out of scope** — the feature belongs to a part of PyTorch that SonicBoom
  does not aim to cover; the note says why.

The lookup matches full names and common spellings (`nn.Linear`, `F.relu`),
and a name typed in the address bar, as in `pytorch.html?q=torch.add`, runs the
same query.

<div class="pt-form">
  <input id="pytorch-query" type="search" placeholder="torch.nn.Linear"
         aria-label="PyTorch name" autocomplete="off">
</div>
<div id="pytorch-results" class="pt-results"></div>

## Names with a SonicBoom counterpart

The table below lists only the names that are **supported** or have a
**recipe**, grouped by the PyTorch module they come from.
