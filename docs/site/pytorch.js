/* The PyTorch lookup: a case-insensitive match over pytorch-map.json.
 * An exact name comes first, then a match on the last segment, then a prefix
 * match, then a substring match. No libraries, no network beyond the one
 * same-origin JSON file. */
(function () {
  "use strict";

  var root = (document.body && document.body.dataset.root) || "";
  var entries = null;
  var loading = false;
  var waiting = [];

  function load(callback) {
    if (entries) {
      callback(entries);
      return;
    }
    waiting.push(callback);
    if (loading) {
      return;
    }
    loading = true;
    fetch(root + "pytorch-map.json")
      .then(function (response) { return response.json(); })
      .then(function (data) {
        entries = data;
        loading = false;
        var callbacks = waiting;
        waiting = [];
        callbacks.forEach(function (one) { one(entries); });
      })
      .catch(function () {
        entries = [];
        loading = false;
        waiting = [];
      });
  }

  function lastSegment(name) {
    var dot = name.lastIndexOf(".");
    return dot < 0 ? name : name.slice(dot + 1);
  }

  /* The score of one spelling of a name against the lower-case query, or -1. */
  function scoreOf(text, query) {
    var lower = text.toLowerCase();
    if (lower === query) {
      return 0;
    }
    if (lastSegment(lower) === query) {
      return 1;
    }
    if (lower.indexOf(query) === 0) {
      return 2;
    }
    if (lower.indexOf(query) >= 0) {
      return 3;
    }
    return -1;
  }

  function score(entry, query) {
    var best = -1;
    [entry.n].concat(entry.a || []).forEach(function (text) {
      var one = scoreOf(text, query);
      if (one >= 0 && (best < 0 || one < best)) {
        best = one;
      }
    });
    return best;
  }

  function search(query) {
    var matches = [];
    entries.forEach(function (entry) {
      var one = score(entry, query);
      if (one >= 0) {
        matches.push([one, entry]);
      }
    });
    matches.sort(function (a, b) {
      if (a[0] !== b[0]) {
        return a[0] - b[0];
      }
      if (a[1].n.length !== b[1].n.length) {
        return a[1].n.length - b[1].n.length;
      }
      return a[1].n < b[1].n ? -1 : a[1].n > b[1].n ? 1 : 0;
    });
    return matches.slice(0, 30).map(function (pair) { return pair[1]; });
  }

  function element(tag, className, text) {
    var node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (text !== undefined) {
      node.textContent = text;
    }
    return node;
  }

  function nameLine(entry) {
    var line = element("div", "pt-name");
    var name = entry.doc ? element("a", "", entry.n) : element("span", "", entry.n);
    if (entry.doc) {
      name.href = entry.doc;
    }
    line.appendChild(name);
    line.appendChild(element("span", "pt-status", entry.st.replace("-", " ")));
    return line;
  }

  function useLine(entry) {
    var line = element("div", "pt-use");
    line.appendChild(document.createTextNode("SonicBoom: "));
    (entry.use || []).forEach(function (use, index) {
      if (index > 0) {
        line.appendChild(document.createTextNode(", "));
      }
      var link = element("a", "", use.symbol);
      link.href = root + use.link;
      line.appendChild(link);
      line.appendChild(element("span", "pt-module", " in " + use.module));
    });
    return line;
  }

  function renderEntry(entry) {
    var box = element("div", "pt-entry");
    box.appendChild(nameLine(entry));
    if (entry.st === "supported") {
      box.appendChild(useLine(entry));
    } else if (entry.st === "recipe") {
      var pre = element("pre", "pt-recipe");
      pre.appendChild(element("code", "", entry.recipe));
      box.appendChild(pre);
    } else if (entry.st === "missing") {
      box.appendChild(element("p", "pt-note", "Not available yet."));
    } else {
      box.appendChild(element("p", "pt-note", "Out of scope."));
    }
    if (entry.note) {
      box.appendChild(element("p", "pt-note", entry.note));
    }
    return box;
  }

  function render(box, results, query) {
    box.textContent = "";
    if (results.length === 0) {
      box.appendChild(element("div", "empty",
                              "No name matches “" + query + "”."));
    }
    results.forEach(function (entry) {
      box.appendChild(renderEntry(entry));
    });
    box.hidden = false;
  }

  function run(box, input) {
    var query = input.value.trim().toLowerCase();
    if (query === "") {
      box.hidden = true;
      return;
    }
    load(function (data) {
      if (data.length === 0) {
        return;
      }
      render(box, search(query), input.value.trim());
    });
  }

  function attach(input) {
    var box = document.getElementById("pytorch-results");
    if (!box) {
      return;
    }
    var timer = null;
    input.addEventListener("input", function () {
      if (input.value.trim() === "") {
        box.hidden = true;
        return;
      }
      if (timer) {
        clearTimeout(timer);
      }
      timer = setTimeout(function () { run(box, input); }, 80);
    });
    input.addEventListener("keydown", function (event) {
      if (event.key === "Escape") {
        box.hidden = true;
        input.blur();
      }
    });
    var query = null;
    try {
      query = new URLSearchParams(location.search).get("q");
    } catch (error) {
      query = null;
    }
    if (query) {
      input.value = query;
      run(box, input);
    }
  }

  var input = document.getElementById("pytorch-query");
  if (input) {
    attach(input);
  }
})();
