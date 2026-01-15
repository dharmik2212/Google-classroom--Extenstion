let activeController = null;

async function getGoogleAccessToken({ interactive }) {
  try {
    const token = await chrome.identity.getAuthToken({ interactive: !!interactive });
    return typeof token === "string" && token.trim() ? token.trim() : null;
  } catch {
    return null;
  }
}

function questionWantsCode(q) {
  const s = String(q || "").toLowerCase();
  return /(code|snippet|example|script|program|python|javascript|js\b|typescript|ts\b|java\b|c\+\+|c#|sql|regex|bash|powershell)/.test(s);
}

function stripFencedCodeBlocks(md) {
  // Remove triple-backtick blocks entirely.
  return String(md || "")
    .replace(/```[\s\S]*?```/g, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function escapeHtml(s) {
  return String(s || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function sanitizeLinkUrl(url) {
  const u = String(url || "").trim();
  if (!u) return null;
  // Only allow safe, explicit schemes.
  if (/^(https?:\/\/|mailto:)/i.test(u)) return u;
  return null;
}

function sanitizeImageUrl(url) {
  const u = String(url || "").trim();
  if (!u) return null;
  // Allow common safe schemes for images.
  if (/^(https?:\/\/)/i.test(u)) return u;
  if (/^data:image\//i.test(u)) return u;
  return null;
}

function renderInlineMarkdown(raw) {
  let s = String(raw || "");

  const images = [];
  s = s.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (_m, alt, url) => {
    const idx = images.length;
    images.push({ alt, url });
    return `@@IMG_${idx}@@`;
  });

  const links = [];
  s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_m, text, url) => {
    const idx = links.length;
    links.push({ text, url });
    return `@@LINK_${idx}@@`;
  });

  const codeSpans = [];
  s = s.replace(/`([^`\n]+)`/g, (_m, code) => {
    const idx = codeSpans.length;
    codeSpans.push(code);
    return `@@CODESPAN_${idx}@@`;
  });

  s = escapeHtml(s);

  // Bold and italics (simple, non-nested).
  s = s.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/_([^_\n]+)_/g, "<em>$1</em>");
  s = s.replace(/\*([^*\n]+)\*/g, "<em>$1</em>");

  s = s.replace(/@@CODESPAN_(\d+)@@/g, (_m, idxStr) => {
    const idx = Number(idxStr);
    const code = idx >= 0 && idx < codeSpans.length ? codeSpans[idx] : "";
    return `<code>${escapeHtml(code)}</code>`;
  });

  s = s.replace(/@@LINK_(\d+)@@/g, (_m, idxStr) => {
    const idx = Number(idxStr);
    const tok = idx >= 0 && idx < links.length ? links[idx] : null;
    if (!tok) return "";

    const href = sanitizeLinkUrl(tok.url);
    const label = escapeHtml(tok.text);
    if (!href) return label;
    return `<a href="${escapeHtml(href)}" target="_blank" rel="noreferrer noopener">${label}</a>`;
  });

  s = s.replace(/@@IMG_(\d+)@@/g, (_m, idxStr) => {
    const idx = Number(idxStr);
    const tok = idx >= 0 && idx < images.length ? images[idx] : null;
    if (!tok) return "";

    const src = sanitizeImageUrl(tok.url);
    const alt = escapeHtml(tok.alt || "");
    if (!src) return alt ? `<em>${alt}</em>` : "";
    return `<img src="${escapeHtml(src)}" alt="${alt}" loading="lazy" referrerpolicy="no-referrer" />`;
  });

  return s;
}

function renderMarkdownToHtml(markdown) {
  const md = String(markdown || "").replace(/\r/g, "");
  const lines = md.split("\n");

  const out = [];
  let inCode = false;
  let codeLang = "";
  let codeLines = [];
  let listType = null; // 'ul' | 'ol'
  let paragraphLines = [];
  let quoteLines = null;

  const splitTableRow = (line) => {
    let s = String(line || "").trim();
    if (s.startsWith("|")) s = s.slice(1);
    if (s.endsWith("|")) s = s.slice(0, -1);
    return s.split("|").map((c) => c.trim());
  };

  const isTableSeparator = (line) => {
    const cells = splitTableRow(line);
    if (cells.length < 2) return false;
    return cells.every((c) => /^:?-{3,}:?$/.test(c.replace(/\s+/g, "")));
  };

  const flushParagraph = () => {
    const text = paragraphLines.join(" ").trim();
    paragraphLines = [];
    if (!text) return;
    out.push(`<p>${renderInlineMarkdown(text)}</p>`);
  };

  const flushList = () => {
    if (!listType) return;
    out.push(`</${listType}>`);
    listType = null;
  };

  const flushQuote = () => {
    if (!quoteLines) return;
    const rendered = quoteLines
      .map((l) => renderInlineMarkdown(l))
      .join("<br/>");
    out.push(`<blockquote>${rendered}</blockquote>`);
    quoteLines = null;
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (inCode) {
      const fence = line.match(/^```\s*$/);
      if (fence) {
        const langClass = codeLang ? ` class="language-${escapeHtml(codeLang)}"` : "";
        out.push(
          `<pre><code${langClass}>${escapeHtml(codeLines.join("\n"))}</code></pre>`
        );
        inCode = false;
        codeLang = "";
        codeLines = [];
      } else {
        codeLines.push(line);
      }
      continue;
    }

    const codeStart = line.match(/^```\s*([a-zA-Z0-9_-]+)?\s*$/);
    if (codeStart) {
      flushParagraph();
      flushList();
      flushQuote();
      inCode = true;
      codeLang = (codeStart[1] || "").trim();
      codeLines = [];
      continue;
    }

    const trimmed = line.trim();
    if (!trimmed) {
      flushParagraph();
      flushList();
      flushQuote();
      continue;
    }

    // Horizontal rule
    if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
      flushParagraph();
      flushList();
      flushQuote();
      out.push("<hr/>");
      continue;
    }

    // GitHub-flavored Markdown table (header + separator + rows)
    if (line.includes("|") && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
      flushParagraph();
      flushList();
      flushQuote();

      const headers = splitTableRow(line);
      i += 1; // skip separator line

      const rows = [];
      while (i + 1 < lines.length) {
        const next = lines[i + 1];
        const nextTrim = (next || "").trim();
        if (!nextTrim) break;
        if (nextTrim.startsWith("```")) break;
        if (!next.includes("|")) break;
        rows.push(splitTableRow(next));
        i += 1;
      }

      const colCount = Math.max(
        headers.length,
        rows.reduce((m, r) => Math.max(m, r.length), 0)
      );

      const normRow = (r) => {
        const outRow = Array(colCount).fill("");
        for (let j = 0; j < Math.min(colCount, r.length); j++) outRow[j] = r[j];
        return outRow;
      };

      const head = normRow(headers);
      out.push("<table>");
      out.push("<thead><tr>" + head.map((c) => `<th>${renderInlineMarkdown(c)}</th>`).join("") + "</tr></thead>");
      out.push("<tbody>");
      for (const r of rows) {
        const rr = normRow(r);
        out.push("<tr>" + rr.map((c) => `<td>${renderInlineMarkdown(c)}</td>`).join("") + "</tr>");
      }
      out.push("</tbody></table>");
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      flushParagraph();
      flushList();
      flushQuote();
      const level = heading[1].length;
      out.push(`<h${level}>${renderInlineMarkdown(heading[2].trim())}</h${level}>`);
      continue;
    }

    const quote = line.match(/^>\s?(.*)$/);
    if (quote) {
      flushParagraph();
      flushList();
      if (!quoteLines) quoteLines = [];
      quoteLines.push(quote[1]);
      continue;
    }
    flushQuote();

    const ul = line.match(/^[-*]\s+(.*)$/);
    if (ul) {
      flushParagraph();
      if (listType && listType !== "ul") flushList();
      if (!listType) {
        listType = "ul";
        out.push("<ul>");
      }
      out.push(`<li>${renderInlineMarkdown(ul[1].trim())}</li>`);
      continue;
    }

    const ol = line.match(/^\d+\.\s+(.*)$/);
    if (ol) {
      flushParagraph();
      if (listType && listType !== "ol") flushList();
      if (!listType) {
        listType = "ol";
        out.push("<ol>");
      }
      out.push(`<li>${renderInlineMarkdown(ol[1].trim())}</li>`);
      continue;
    }

    if (listType) flushList();
    paragraphLines.push(trimmed);
  }

  if (inCode) {
    const langClass = codeLang ? ` class="language-${escapeHtml(codeLang)}"` : "";
    out.push(
      `<pre><code${langClass}>${escapeHtml(codeLines.join("\n"))}</code></pre>`
    );
  }
  flushParagraph();
  flushList();
  flushQuote();

  return out.join("\n");
}

// UX: Enter submits; Shift+Enter makes a newline.
(() => {
  const questionEl = document.getElementById("question");
  const askBtnEl = document.getElementById("ask");
  if (!questionEl || !askBtnEl) return;

  questionEl.addEventListener("keydown", (e) => {
    if (e.isComposing) return;
    if (e.key !== "Enter") return;
    if (e.shiftKey) return; // newline
    e.preventDefault();
    askBtnEl.click();
  });
})();

document.getElementById("ask").addEventListener("click", async () => {
  const MAX_CONTEXT_CHARS = 12000;
  const shrinkContext = (s) => {
    // Preserve newlines (important for tables/code), but normalize spaces.
    const raw = String(s || "").replace(/\r/g, "");
    const normalized = raw
      .split("\n")
      .map((line) => line.replace(/[\t ]+/g, " ").trimEnd())
      .join("\n")
      .replace(/\n{3,}/g, "\n\n")
      .trim();
    return normalized.slice(0, MAX_CONTEXT_CHARS);
  };

  const askBtn = document.getElementById("ask");
  const answerDiv = document.getElementById("answer");
  const question = (document.getElementById("question").value || "").trim();

  if (!question) {
    answerDiv.innerText = "Type a question first.";
    return;
  }

  if (activeController) {
    try { activeController.abort(); } catch {}
  }
  activeController = new AbortController();

  askBtn.disabled = true;
  answerDiv.dataset.mode = "text";
  answerDiv.textContent = "Reading page...";

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    args: [question],
    func: (questionArg) => {
      const normalizeSpacesKeepNewlines = (s) =>
        String(s || "")
          .replace(/\r/g, "")
          .replace(/[\t ]+/g, " ")
          .replace(/\n{3,}/g, "\n\n")
          .trim();

      const normalizeAllWs = (s) =>
        String(s || "")
          .replace(/\s+/g, " ")
          .trim();

      const rawQuestion = normalizeAllWs(questionArg);

      // Prefer user selection when available.
      let selectionText = "";
      try {
        const sel = window.getSelection?.();
        selectionText = normalizeSpacesKeepNewlines(sel ? sel.toString() : "");
      } catch {
        selectionText = "";
      }

      const MAX_SCAN_CHARS = 60000;
      const rawText = normalizeSpacesKeepNewlines(document.body ? document.body.innerText : "");
      const scanned = rawText.slice(0, MAX_SCAN_CHARS);

      const findDriveFileId = () => {
        const urls = [];
        const push = (u) => {
          if (!u) return;
          const s = String(u);
          if (s && s.length < 5000) urls.push(s);
        };

        try {
          for (const el of Array.from(document.querySelectorAll("a[href]"))) push(el.getAttribute("href"));
          for (const el of Array.from(document.querySelectorAll("iframe[src]"))) push(el.getAttribute("src"));
          for (const el of Array.from(document.querySelectorAll("embed[src]"))) push(el.getAttribute("src"));
          for (const el of Array.from(document.querySelectorAll("object[data]"))) push(el.getAttribute("data"));
          for (const el of Array.from(document.querySelectorAll("link[rel='canonical'][href]"))) push(el.getAttribute("href"));
        } catch {
          // ignore
        }

        const patterns = [
          /https?:\/\/drive\.google\.com\/file\/d\/([a-zA-Z0-9_-]{10,})/,
          /https?:\/\/docs\.google\.com\/document\/d\/([a-zA-Z0-9_-]{10,})/,
          /https?:\/\/docs\.google\.com\/viewer\?[^\s#]*?\b(?:id|fileid)=([a-zA-Z0-9_-]{10,})/i,
          /\b(?:id|fileid)=([a-zA-Z0-9_-]{10,})/i,
          /\/d\/([a-zA-Z0-9_-]{10,})/,
        ];

        const parseCandidate = (u) => {
          if (!u) return null;
          let url = String(u).trim();
          if (!url) return null;
          try {
            url = new URL(url, location.href).toString();
          } catch {
            // keep as-is
          }
          // unwrap docs viewer url=... param
          try {
            const parsed = new URL(url);
            const inner = parsed.searchParams.get("url") || parsed.searchParams.get("q");
            if (inner && inner.length > 10) url = inner;
          } catch {
            // ignore
          }

          for (const re of patterns) {
            const m = url.match(re);
            if (m && m[1]) return { fileId: m[1], sourceUrl: url };
          }
          return null;
        };

        for (const u of urls) {
          const cand = parseCandidate(u);
          if (cand) return cand;
        }
        return null;
      };

      const docRef = findDriveFileId();

      // If selection is meaningful, just use it.
      if (selectionText.length >= 40) {
        const images = Array.from(document.images)
          .map((img) => ({
            url: img.currentSrc || img.src,
            alt: normalizeAllWs(img.alt || ""),
            w: img.naturalWidth || 0,
            h: img.naturalHeight || 0,
          }))
          .filter((x) => x.url)
          .sort((a, b) => (b.w * b.h) - (a.w * a.h))
          .slice(0, 6);

        const figureLines = images
          .filter((x) => x.w * x.h >= 120 * 120)
          .map((x) => `- ${x.alt || "(no alt)"} (${x.w}x${x.h}) ${x.url}`)
          .join("\n");

        const text = figureLines
          ? `Figures/Diagrams on the page:\n${figureLines}\n\n${selectionText}`
          : selectionText;

        return { text, images: images.map((x) => x.url), pageUrl: location.href, docRef };
      }

      const stop = new Set([
        "the","a","an","and","or","but","to","of","in","on","for","with","as","at","by",
        "is","are","was","were","be","been","it","this","that","these","those","from",
        "you","your","we","our","they","their","i","me","my"
      ]);

      const tokenize = (s) =>
        normalizeAllWs(s)
          .toLowerCase()
          .replace(/[^a-z0-9\s]/g, " ")
          .split(/\s+/)
          .filter((w) => w && w.length >= 2 && !stop.has(w));

      const qTokens = Array.from(new Set(tokenize(rawQuestion)));

      // Paragraph-ish chunks; fallback to slicing.
      let chunks = scanned
        .split(/\n{2,}/)
        .map((p) => normalizeSpacesKeepNewlines(p))
        .filter((p) => p.length >= 30);

      if (chunks.length < 3) {
        chunks = [];
        for (let i = 0; i < scanned.length; i += 700) {
          const part = normalizeAllWs(scanned.slice(i, i + 700));
          if (part) chunks.push(part);
        }
      }

      const scored = chunks.map((text, idx) => {
        const lower = text.toLowerCase();
        let score = 0;
        for (const tok of qTokens) {
          if (lower.includes(tok)) score += 1;
        }
        const qPhrase = rawQuestion.toLowerCase();
        if (qPhrase && qPhrase.length >= 12 && lower.includes(qPhrase)) score += 3;
        return { idx, text, score };
      });

      const picked = scored
        .sort((a, b) => b.score - a.score || a.idx - b.idx)
        .slice(0, 10)
        .filter((c) => c.score > 0);

      const finalPicked = (picked.length ? picked : scored.slice(0, 4))
        .sort((a, b) => a.idx - b.idx);

      const escapePipes = (s) => String(s || "").replace(/\|/g, "\\|");
      const extractTablesMarkdown = () => {
        const tables = Array.from(document.querySelectorAll("table")).slice(0, 2);
        const parts = [];
        for (const tbl of tables) {
          const rows = Array.from(tbl.querySelectorAll("tr")).slice(0, 10);
          const grid = rows
            .map((tr) =>
              Array.from(tr.querySelectorAll("th,td"))
                .slice(0, 8)
                .map((cell) => normalizeAllWs(cell.innerText))
            )
            .filter((r) => r.length >= 2);

          if (grid.length < 2) continue;
          const cols = Math.max(...grid.map((r) => r.length));
          if (cols < 2) continue;

          const header = (grid[0] || []).concat(Array(cols).fill("")).slice(0, cols);
          const headerLine = `| ${header.map((c) => escapePipes(c)).join(" | ")} |`;
          const sepLine = `| ${Array(cols).fill("---").join(" | ")} |`;
          const bodyLines = grid.slice(1).map((r) => {
            const row = r.concat(Array(cols).fill("")).slice(0, cols);
            return `| ${row.map((c) => escapePipes(c)).join(" | ")} |`;
          });

          parts.push([headerLine, sepLine, ...bodyLines].join("\n"));
        }
        return parts.join("\n\n");
      };

      const MAX_SELECTED_CHARS = 12000;
      let selected = "";
      for (const c of finalPicked) {
        if (selected.length >= MAX_SELECTED_CHARS) break;
        const remaining = MAX_SELECTED_CHARS - selected.length;
        const chunkText = c.text.slice(0, remaining);
        if (!chunkText) continue;
        selected += (selected ? "\n\n" : "") + chunkText;
      }

      const tablesMd = extractTablesMarkdown();

      const bestImages = Array.from(document.images)
        .map((img) => ({
          url: img.currentSrc || img.src,
          alt: normalizeAllWs(img.alt || ""),
          w: img.naturalWidth || 0,
          h: img.naturalHeight || 0,
        }))
        .filter((x) => x.url)
        .sort((a, b) => (b.w * b.h) - (a.w * a.h))
        .slice(0, 6);

      const figureLines = bestImages
        .filter((x) => x.w * x.h >= 120 * 120)
        .map((x) => `- ${x.alt || "(no alt)"} (${x.w}x${x.h}) ${x.url}`)
        .join("\n");

      const prefixes = [];
      if (tablesMd) {
        prefixes.push(`Tables extracted from the page (Markdown):\n${tablesMd}`);
      }
      if (figureLines) {
        prefixes.push(`Figures/Diagrams on the page (URLs + alt text):\n${figureLines}`);
      }
      if (prefixes.length) {
        selected = (prefixes.join("\n\n") + "\n\n" + selected).slice(0, MAX_SELECTED_CHARS);
      }

      const images = bestImages.map((x) => x.url);
      const pageUrl = location.href;

      return { text: selected, images, pageUrl, docRef };
    }
  }, async (results) => {

    if (!results || !results[0] || !results[0].result) {
      askBtn.disabled = false;
      answerDiv.innerText = "Could not read the page.";
      return;
    }

    const payload = {
      question: question,
      context: shrinkContext(results[0].result.text),
      images: results[0].result.images,
      pageUrl: results[0].result.pageUrl,
      docRef: results[0].result.docRef || null,
    };

    // If we detected a Drive/Docs file id, attempt to grab a Google access token.
    let googleToken = null;
    if (payload.docRef && payload.docRef.fileId) {
      googleToken = await getGoogleAccessToken({ interactive: false });
      if (!googleToken) {
        // Prompt only when user is clearly trying to ask about an open doc.
        googleToken = await getGoogleAccessToken({ interactive: true });
      }
    }

    try {
      answerDiv.dataset.mode = "text";
      answerDiv.textContent = "";
      let streamedText = "";
      let gotDone = false;
      let gotError = false;
      let errorText = "";

      const res = await fetch("http://localhost:8000/ask_stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(googleToken ? { Authorization: `Bearer ${googleToken}` } : {}),
        },
        body: JSON.stringify(payload),
        signal: activeController.signal
      });

      if (!res.ok) {
        const raw = await res.text();
        answerDiv.innerText = raw || `Request failed (${res.status})`;
        return;
      }

      if (!res.body) {
        // Fallback if streaming isn't available.
        const raw = await res.text();
        answerDiv.innerText = raw || "No response body";
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      const processFrames = (frames) => {
        for (const part of frames) {
          const lines = part.split("\n");
          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed || trimmed.startsWith(":")) continue;
            if (!trimmed.startsWith("data:")) continue;

            const dataStr = trimmed.slice(5).trim();
            if (!dataStr) continue;

            let msg = null;
            try {
              msg = JSON.parse(dataStr);
            } catch {
              msg = null;
            }

            if (!msg) continue;

            if (msg.type === "token" && typeof msg.text === "string") {
              streamedText += msg.text;
              answerDiv.dataset.mode = "text";
              answerDiv.textContent = streamedText;
            } else if (msg.type === "error") {
              const parts = [];
              if (typeof msg.message === "string" && msg.message.trim()) {
                parts.push(msg.message.trim());
              }
              if (typeof msg.error === "string" && msg.error.trim()) {
                parts.push(msg.error.trim());
              } else if (msg.error != null) {
                try {
                  parts.push(typeof msg.error === "object" ? JSON.stringify(msg.error) : String(msg.error));
                } catch {
                  parts.push(String(msg.error));
                }
              }
              answerDiv.dataset.mode = "text";
              errorText = parts.join("\n") || "server error please try again";
              answerDiv.textContent = errorText;
              gotError = true;
              return true; // stop
            } else if (msg.type === "done") {
              gotDone = true;
              return true; // stop
            }
          }
        }
        return false;
      };

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        // Be tolerant of CRLF or split CRLF across chunks.
        buffer = buffer.replace(/\r/g, "");

        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";

        const shouldStop = processFrames(parts);
        if (shouldStop) break;
      }

      // Flush any remaining buffered frame.
      buffer += decoder.decode();
      buffer = buffer.replace(/\r/g, "");
      const leftover = buffer.trim();
      if (leftover) {
        processFrames([leftover]);
      }

      // If the stream ended without a done marker (network interruption / popup lifecycle),
      // fetch the full answer via the non-streaming endpoint.
      if (!gotDone && !gotError && streamedText.trim()) {
        try {
          const fullRes = await fetch("http://localhost:8000/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
            signal: activeController.signal,
          });
          if (fullRes.ok) {
            const data = await fullRes.json();
            if (data && typeof data.answer === "string" && data.answer.trim()) {
              streamedText = data.answer;
              gotDone = true;
            }
          }
        } catch {
          // ignore fallback errors; we will render whatever we have.
        }
      }

      if (!gotError) {
        // Render Markdown once at the end for a stable, polished display.
        answerDiv.dataset.mode = "html";
        const finalText = questionWantsCode(question)
          ? streamedText
          : stripFencedCodeBlocks(streamedText);
        answerDiv.innerHTML = renderMarkdownToHtml(finalText);
      }
    } catch (e) {
      if (e && e.name === "AbortError") {
        answerDiv.dataset.mode = "text";
        answerDiv.textContent = "";
      } else {
        answerDiv.dataset.mode = "text";
        answerDiv.textContent = `Network/CORS error: ${e && e.message ? e.message : e}`;
      }
    } finally {
      askBtn.disabled = false;
    }
  });
});