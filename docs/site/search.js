/* Site search: a case-insensitive substring match over search-index.json.
 * Exact names come first, then prefix matches, then substring matches.
 * No libraries, no network beyond the one same-origin JSON file. */

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
    fetch(root + "search-index.json")
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

  function rank(entry, query) {
    var name = entry.t.toLowerCase();
    if (name === query) {
      return 0;
    }
    if (name.indexOf(query) === 0) {
      return 1;
    }
    if (name.indexOf(query) >= 0) {
      return 2;
    }
    if (entry.s && entry.s.toLowerCase().indexOf(query) >= 0) {
      return 3;
    }
    return -1;
  }

  function search(query) {
    var matches = [];
    entries.forEach(function (entry) {
      var score = rank(entry, query);
      if (score >= 0) {
        matches.push([score, entry]);
      }
    });
    matches.sort(function (a, b) {
      if (a[0] !== b[0]) {
        return a[0] - b[0];
      }
      return a[1].t.toLowerCase() < b[1].t.toLowerCase() ? -1 : 1;
    });
    return matches.slice(0, 20).map(function (pair) { return pair[1]; });
  }

  function render(box, results, query) {
    box.textContent = "";
    if (results.length === 0) {
      var empty = document.createElement("div");
      empty.className = "empty";
      empty.textContent = "No matches for “" + query + "”.";
      box.appendChild(empty);
    }
    results.forEach(function (entry) {
      var link = document.createElement("a");
      link.href = root + entry.u;
      var title = document.createElement("span");
      title.textContent = entry.t;
      link.appendChild(title);
      var where = document.createElement("div");
      where.className = "where";
      where.textContent = entry.k + " · " + entry.m;
      link.appendChild(where);
      box.appendChild(link);
    });
    box.hidden = false;
  }

  function attach(input) {
    var box = input.parentElement.querySelector(".search-results");
    if (!box) {
      return;
    }
    var timer = null;
    input.addEventListener("input", function () {
      var query = input.value.trim();
      if (query === "") {
        box.hidden = true;
        return;
      }
      if (timer) {
        clearTimeout(timer);
      }
      timer = setTimeout(function () {
        load(function (data) {
          if (data.length === 0) {
            return;
          }
          render(box, search(query.toLowerCase()), query);
        });
      }, 80);
    });
    input.addEventListener("focus", function () {
      if (input.value.trim() !== "") {
        load(function () {});
      }
    });
    input.addEventListener("keydown", function (event) {
      if (event.key === "Escape") {
        box.hidden = true;
        input.blur();
      }
    });
    document.addEventListener("click", function (event) {
      if (!input.parentElement.contains(event.target)) {
        box.hidden = true;
      }
    });
  }

  document.querySelectorAll(".site-search-input").forEach(attach);
})();
