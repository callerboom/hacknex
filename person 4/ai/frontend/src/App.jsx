import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Microphone, MicrophoneSlash, PaperPlaneRight, Image as ImageIcon, Trash,
  Cpu, HardDrives, ShieldCheck, Clock, Sparkle, X, Gear,
  SunHorizon, Moon, ChatCircleDots, ArrowsClockwise, Check,
  SpeakerHigh, Lightning, Waveform, Robot, User, Copy,
  FileDoc, FilePdf, DownloadSimple, Stop, Headset
} from '@phosphor-icons/react';

/* ─── Audio Helper: Encode Float32 PCM to 16kHz Mono WAV Blob ─── */
function encodeWAV(samples, sampleRate = 16000) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  function writeString(v, offset, str) {
    for (let i = 0; i < str.length; i++) {
      v.setUint8(offset + i, str.charCodeAt(i));
    }
  }
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return new Blob([view], { type: 'audio/wav' });
}

/* ─── Markdown Inline Tokenizer ─── */
function renderInline(str, darkMode = true) {
  if (!str) return '';
  const parts = [];
  // Handles: **bold**, *italic*, `code`, [link](url)
  const regex = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|\[([^\]]+)\]\(([^)]+)\))/g;
  let lastIdx = 0;
  let match;

  while ((match = regex.exec(str)) !== null) {
    if (match.index > lastIdx) {
      parts.push(str.substring(lastIdx, match.index));
    }
    const token = match[0];
    if (token.startsWith('`') && token.endsWith('`')) {
      parts.push(
        <code
          key={match.index}
          className={`px-1.5 py-0.5 mx-0.5 rounded border font-mono text-[11px] sm:text-xs ${
            darkMode
              ? 'bg-violet-950/80 border-violet-500/30 text-cyan-300'
              : 'bg-violet-50 border-violet-200 text-violet-700'
          }`}
        >
          {token.slice(1, -1)}
        </code>
      );
    } else if (token.startsWith('**') && token.endsWith('**')) {
      parts.push(
        <strong key={match.index} className={`font-semibold ${darkMode ? 'text-slate-100' : 'text-slate-800'}`}>
          {token.slice(2, -2)}
        </strong>
      );
    } else if (token.startsWith('*') && token.endsWith('*')) {
      parts.push(
        <em key={match.index} className={`italic ${darkMode ? 'text-slate-300' : 'text-slate-600'}`}>
          {token.slice(1, -1)}
        </em>
      );
    } else if (token.startsWith('[')) {
      // markdown link [text](url)
      parts.push(
        <a key={match.index} href={match[3]} target="_blank" rel="noopener noreferrer"
          className="text-cyan-400 hover:text-cyan-300 underline underline-offset-2 transition-colors">
          {match[2]}
        </a>
      );
    }
    lastIdx = match.index + token.length;
  }

  if (lastIdx < str.length) {
    parts.push(str.substring(lastIdx));
  }

  return parts.length > 0 ? parts : str;
}

/* ─── Sleek Code Block with Syntax Bar & Copy ─── */
function CodeBlock({ language, code, darkMode }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="my-3 rounded-xl overflow-hidden border border-white/[0.07] bg-[#07080f] text-left shadow-sm">
      <div className="flex items-center justify-between px-4 py-2 bg-white/[0.03] border-b border-white/[0.06]">
        <span className="font-mono text-[10px] font-semibold text-violet-400 uppercase tracking-widest">
          {language || 'code'}
        </span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.06] text-slate-400 hover:text-slate-200 transition-all duration-150 text-[11px] font-medium cursor-pointer"
        >
          {copied ? (
            <>
              <Check className="w-3 h-3 text-emerald-400" />
              <span className="text-emerald-400">Copied!</span>
            </>
          ) : (
            <>
              <Copy className="w-3 h-3" />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>
      <pre className="p-4 overflow-x-auto font-mono text-xs sm:text-[12.5px] text-slate-200 leading-relaxed whitespace-pre selection:bg-violet-600/30">
        <code>{code}</code>
      </pre>
    </div>
  );
}

/* ─── Text & Paragraph Parser (Preserves Headings, Lists, Newlines) ─── */
function TextSegment({ text = '', darkMode = true }) {
  const safeText = String(text || '');

  // ── Pre-process: merge lines where a number-dot appears alone on a line
  // e.g. "3.\nNatural Language Processing" → "3. Natural Language Processing"
  const merged = safeText.replace(/^(\d+\.)\s*\n+([^\n])/gm, '$1 $2');

  // ── Group consecutive list items so we can wrap them in a semantic list
  const lines = merged.split('\n');
  const nodes = [];
  let listBuffer = []; // accumulates consecutive list lines
  let prevWasList = false;

  const flushList = (key) => {
    if (listBuffer.length === 0) return;
    nodes.push(
      <div key={`list-${key}`} className="flex flex-col gap-0.5 my-1 pl-1">
        {listBuffer.map(({ prefix, content, idx }) => (
          <div key={idx} className="flex items-start gap-2">
            <span className={`font-mono select-none font-bold text-[11px] pt-0.5 shrink-0 ${
              darkMode ? 'text-violet-400' : 'text-violet-600'
            }`}>
              {prefix}
            </span>
            <span className={`text-[13px] leading-relaxed flex-1 ${
              darkMode ? 'text-slate-200' : 'text-slate-700'
            }`}>
              {renderInline(content, darkMode)}
            </span>
          </div>
        ))}
      </div>
    );
    listBuffer = [];
  };

  lines.forEach((line, lIdx) => {
    const trimmed = line.trim();

    // Empty line — flush list buffer
    if (!trimmed) {
      if (prevWasList) flushList(lIdx);
      nodes.push(<div key={lIdx} className="h-1" />);
      prevWasList = false;
      return;
    }

    // Headings
    if (trimmed.startsWith('### ')) {
      if (prevWasList) flushList(lIdx);
      prevWasList = false;
      nodes.push(
        <h4 key={lIdx} className={`text-[13px] font-bold mt-2 mb-0.5 ${
          darkMode ? 'text-cyan-300' : 'text-cyan-700'
        }`}>
          {renderInline(trimmed.replace(/^###\s+/, ''), darkMode)}
        </h4>
      );
      return;
    }
    if (trimmed.startsWith('## ')) {
      if (prevWasList) flushList(lIdx);
      prevWasList = false;
      nodes.push(
        <h3 key={lIdx} className={`text-[14px] font-bold mt-2 mb-1 ${
          darkMode ? 'text-violet-300' : 'text-violet-700'
        }`}>
          {renderInline(trimmed.replace(/^##\s+/, ''), darkMode)}
        </h3>
      );
      return;
    }
    if (trimmed.startsWith('# ')) {
      if (prevWasList) flushList(lIdx);
      prevWasList = false;
      nodes.push(
        <h2 key={lIdx} className="text-[15px] font-extrabold grad-text mt-3 mb-1">
          {renderInline(trimmed.replace(/^#\s+/, ''), darkMode)}
        </h2>
      );
      return;
    }

    // Horizontal rule
    if (/^[-_*]{3,}$/.test(trimmed)) {
      if (prevWasList) flushList(lIdx);
      prevWasList = false;
      nodes.push(<hr key={lIdx} className={`my-2 border-t ${
        darkMode ? 'border-white/10' : 'border-slate-200'
      }`} />);
      return;
    }

    // Numbered list: "1. text" or "1.text" (no space sometimes from LLM)
    const numberedMatch = trimmed.match(/^(\d+)\.\s+(.*)/s);
    // Bullet list: "- text" or "* text"
    const bulletMatch = trimmed.match(/^[-*]\s+(.*)/s);

    // Standalone number dot (e.g. the LLM output "3." on its own line — show as numbered prefix with empty content)
    const standaloneNum = !numberedMatch && !bulletMatch && /^\d+\.$/.test(trimmed);

    if (numberedMatch) {
      listBuffer.push({ prefix: `${numberedMatch[1]}.`, content: numberedMatch[2], idx: lIdx });
      prevWasList = true;
      return;
    }
    if (bulletMatch) {
      listBuffer.push({ prefix: '•', content: bulletMatch[1], idx: lIdx });
      prevWasList = true;
      return;
    }
    if (standaloneNum) {
      // Orphaned "3." — treat it as an empty list item, it'll merge with next line content via the pre-process step
      // If it still shows up here, skip it silently (the pre-process merged it already)
      return;
    }

    // Continuation line indented under a list item (starts with spaces/tabs)
    if (prevWasList && line.match(/^\s{2,}/) && listBuffer.length > 0) {
      // Append to last list item
      const last = listBuffer[listBuffer.length - 1];
      last.content = last.content + ' ' + trimmed;
      return;
    }

    // Regular paragraph — flush pending list first
    if (prevWasList) flushList(lIdx);
    prevWasList = false;

    nodes.push(
      <p key={lIdx} className={`text-[13px] leading-relaxed whitespace-pre-wrap ${
        darkMode ? 'text-slate-200' : 'text-slate-700'
      }`}>
        {renderInline(line, darkMode)}
      </p>
    );
  });

  // Flush any trailing list
  if (prevWasList) flushList('end');

  return (
    <div className="space-y-0.5 text-left">
      {nodes}
    </div>
  );
}

/* ─── Complete Formatted Message Component ─── */
function FormattedMessage({ content = '', isUser = false, darkMode = true }) {
  const safeContent = String(content || '');
  if (isUser) {
    return <div className="whitespace-pre-wrap break-words text-left">{safeContent}</div>;
  }

  const parts = [];
  const codeBlockRegex = /```([a-zA-Z0-9_-]*)\n?([\s\S]*?)```/g;
  let lastIndex = 0;
  let match;

  while ((match = codeBlockRegex.exec(safeContent)) !== null) {
    if (match.index > lastIndex) {
      parts.push({
        type: 'text',
        content: safeContent.substring(lastIndex, match.index),
      });
    }
    parts.push({
      type: 'code',
      language: match[1]?.trim() || 'code',
      code: match[2]?.trimEnd() || '',
    });
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < safeContent.length) {
    parts.push({
      type: 'text',
      content: safeContent.substring(lastIndex),
    });
  }

  return (
    <div className="space-y-1 break-words">
      {parts.map((part, idx) => {
        if (part.type === 'code') {
          return <CodeBlock key={idx} language={part.language} code={part.code} darkMode={darkMode} />;
        }
        return <TextSegment key={idx} text={part.content} darkMode={darkMode} />;
      })}
    </div>
  );
}


/* ─── Neural-Pulse Loading Component ─── */
function NeuralLoader({ label = 'Processing…' }) {
  return (
    <div className="flex items-end gap-[3px] h-6">
      {[0, 1, 2, 3, 4].map((i) => (
        <div
          key={i}
          className="audio-bar w-[2.5px] rounded-full bg-gradient-to-t from-violet-500 to-cyan-400"
          style={{ animationDelay: `${i * 0.14}s`, height: '3px' }}
        />
      ))}
      <span className="ml-2.5 text-[11px] shimmer-text font-mono tracking-tight">{label}</span>
    </div>
  );
}

/* ─── Settings Modal ─── */
function SettingsModal({ onClose, darkMode, onToggleDark }) {
  return (
    <div className="settings-overlay" onClick={onClose}>
      <div
        className="settings-panel glass-glow"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-5">
          <div>
            <h2 className="font-semibold text-[15px] text-white tracking-tight">Settings</h2>
            <p className="text-[11px] text-slate-500 mt-0.5">EdgeVoice AI preferences</p>
          </div>
          <button
            onClick={onClose}
            className="w-7 h-7 rounded-lg flex items-center justify-center hover:bg-white/[0.07] text-slate-400 hover:text-slate-200 transition-all duration-150"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Appearance */}
        <div className="mb-4">
          <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-600 mb-2.5 px-0.5">Appearance</p>
          <div className={`flex items-center justify-between p-3.5 rounded-xl border ${
            darkMode ? 'border-white/[0.06] bg-white/[0.025]' : 'border-violet-100 bg-violet-50/50'
          }`}>
            <div className="flex items-center gap-3">
              <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                darkMode ? 'bg-violet-500/15 border border-violet-500/20' : 'bg-amber-50 border border-amber-200'
              }`}>
                {darkMode ? <Moon className="w-4 h-4 text-violet-400" /> : <SunHorizon className="w-4 h-4 text-amber-500" />}
              </div>
              <div>
                <p className={`text-[13px] font-medium ${darkMode ? 'text-slate-200' : 'text-slate-700'}`}>
                  {darkMode ? 'Dark Mode' : 'Light Mode'}
                </p>
                <p className={`text-[11px] ${darkMode ? 'text-slate-500' : 'text-slate-400'}`}>Switch interface theme</p>
              </div>
            </div>
            <button
              onClick={onToggleDark}
              className={`w-11 h-6 rounded-full relative transition-colors duration-250 ${darkMode ? 'bg-violet-600' : 'bg-slate-300'}`}
            >
              <span className={`absolute top-1 w-4 h-4 bg-white rounded-full shadow transition-all duration-250 ${darkMode ? 'left-6' : 'left-1'}`} />
            </button>
          </div>
        </div>

        {/* System */}
        <div className="mb-4">
          <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-600 mb-2.5 px-0.5">System</p>
          <div className={`rounded-xl border overflow-hidden ${
            darkMode ? 'border-white/[0.06]' : 'border-violet-100'
          }`}>
            {[
              { label: 'Model', value: 'Qwen 2.5 · 1.5B' },
              { label: 'Mode', value: '100% Offline · CPU Only' },
              { label: 'Cores', value: '4 Pinned · AMD Ryzen 7' },
              { label: 'RAM Cap', value: '3,072 MB' },
            ].map(({ label, value }, i, arr) => (
              <div
                key={label}
                className={`flex items-center justify-between px-3.5 py-2.5 ${
                  i < arr.length - 1
                    ? darkMode ? 'border-b border-white/[0.045]' : 'border-b border-violet-50'
                    : ''
                } ${
                  darkMode ? 'bg-white/[0.02]' : 'bg-white'
                }`}
              >
                <span className={`text-[12px] ${darkMode ? 'text-slate-400' : 'text-slate-500'}`}>{label}</span>
                <span className={`font-mono text-[11px] font-medium ${
                  darkMode ? 'text-slate-200' : 'text-violet-700'
                }`}>{value}</span>
              </div>
            ))}
          </div>
        </div>

        <p className={`text-center text-[10px] font-mono ${
          darkMode ? 'text-slate-700' : 'text-slate-400'
        }`}>HNX26EPS08 · EdgeVoice OS</p>
      </div>
    </div>
  );
}

/* ─── Document & OCR Extracted Content Card ─── */
function ExtractedTextCard({ message, onSolveWithAI, onSummarizeWithAI, darkMode }) {
  const [copied, setCopied] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);

  const cleanText = message.cleanText || message.text || '';
  const isPdf = message.fileType === 'pdf' || (message.wordFilename && message.wordFilename.endsWith('.docx') && !message.imageUrl);
  const wordUrl = message.wordUrl;
  const wordFilename = message.wordFilename || (isPdf ? 'converted_document.docx' : 'extracted_text.docx');

  const handleCopy = () => {
    navigator.clipboard.writeText(cleanText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={`mt-3 rounded-xl border p-3.5 text-left ${
      darkMode 
        ? 'bg-[#0c0d18] border-white/[0.08]' 
        : 'bg-white border-violet-100 shadow-sm'
    }`}>
      {/* Header */}
      <div className={`flex items-center justify-between gap-3 mb-3 pb-3 border-b ${
        darkMode ? 'border-white/[0.06]' : 'border-violet-50'
      }`}>
        <div className="flex items-center gap-2.5 min-w-0">
          <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${
            isPdf
              ? darkMode ? 'bg-rose-500/15 text-rose-400 border border-rose-500/20' : 'bg-rose-50 text-rose-500 border border-rose-200'
              : darkMode ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/20' : 'bg-cyan-50 text-cyan-600 border border-cyan-200'
          }`}>
            {isPdf ? <FilePdf weight="fill" className="w-4 h-4" /> : <FileDoc weight="fill" className="w-4 h-4" />}
          </div>
          <div className="min-w-0">
            <p className={`text-[12px] font-semibold truncate ${
              darkMode ? 'text-slate-200' : 'text-slate-700'
            }`}>
              {message.filename || wordFilename}
            </p>
            <p className={`text-[10px] font-mono mt-0.5 ${
              darkMode ? 'text-slate-500' : 'text-slate-400'
            }`}>
              {isPdf 
                ? `${message.pageCount || 1} page${(message.pageCount || 1) !== 1 ? 's' : ''} · Word (.docx)`
                : 'RapidOCR · Text Extraction'}
            </p>
          </div>
        </div>

        {wordUrl && (
          <span className="shrink-0 text-[10px] px-2 py-0.5 rounded-md bg-emerald-500/12 border border-emerald-500/25 text-emerald-400 font-semibold font-mono">
            Ready
          </span>
        )}
      </div>

      {/* Text preview */}
      <div className="relative mb-3">
        <div className={`text-[11px] font-mono leading-relaxed p-3 rounded-lg border overflow-y-auto whitespace-pre-wrap transition-all ${
          isExpanded ? 'max-h-80' : 'max-h-32'
        } ${
          darkMode
            ? 'bg-black/50 border-white/[0.05] text-slate-300'
            : 'bg-slate-50 border-slate-100 text-slate-600'
        }`}>
          {cleanText || 'No text extracted.'}
        </div>
        {cleanText.length > 180 && (
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className={`mt-1.5 text-[11px] font-medium cursor-pointer ${
              darkMode ? 'text-violet-400 hover:text-violet-300' : 'text-violet-500 hover:text-violet-700'
            }`}
          >
            {isExpanded ? '↑ Show less' : '↓ Show all extracted content'}
          </button>
        )}
      </div>

      {/* Actions */}
      <div className="flex flex-wrap items-center gap-1.5">
        <button
          onClick={handleCopy}
          className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[11px] font-medium border transition-all duration-150 cursor-pointer active:scale-95 ${
            darkMode
              ? 'bg-white/[0.04] hover:bg-violet-500/12 border-white/[0.07] hover:border-violet-500/25 text-slate-400 hover:text-violet-300'
              : 'bg-white hover:bg-violet-50 border-violet-100 text-slate-500 hover:text-violet-600'
          }`}
        >
          {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
          <span>{copied ? 'Copied!' : 'Copy Text'}</span>
        </button>

        {wordUrl && (
          <a
            href={wordUrl}
            download={wordFilename}
            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[11px] font-medium border transition-all duration-150 cursor-pointer active:scale-95 ${
              darkMode
                ? 'bg-cyan-500/10 hover:bg-cyan-500/18 border-cyan-500/20 hover:border-cyan-500/35 text-cyan-400 hover:text-cyan-300'
                : 'bg-cyan-50 hover:bg-cyan-100 border-cyan-200 text-cyan-600'
            }`}
          >
            <DownloadSimple weight="bold" className="w-3 h-3" />
            <span>Word (.docx)</span>
          </a>
        )}

        {isPdf ? (
          <button
            onClick={() => onSummarizeWithAI && onSummarizeWithAI(cleanText)}
            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[11px] font-medium border transition-all duration-150 cursor-pointer active:scale-95 ${
              darkMode
                ? 'bg-amber-500/10 hover:bg-amber-500/18 border-amber-500/20 text-amber-400 hover:text-amber-300'
                : 'bg-amber-50 hover:bg-amber-100 border-amber-200 text-amber-600'
            }`}
          >
            <Sparkle weight="fill" className="w-3 h-3" />
            <span>Summarize</span>
          </button>
        ) : (
          <button
            onClick={() => onSolveWithAI && onSolveWithAI(cleanText)}
            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[11px] font-medium border transition-all duration-150 cursor-pointer active:scale-95 ${
              darkMode
                ? 'bg-violet-500/10 hover:bg-violet-500/18 border-violet-500/20 text-violet-400 hover:text-violet-300'
                : 'bg-violet-50 hover:bg-violet-100 border-violet-200 text-violet-600'
            }`}
          >
            <Lightning weight="fill" className="w-3 h-3" />
            <span>Solve with AI</span>
          </button>
        )}
      </div>
    </div>
  );
}

/* ─── Desktop Screenshot Preview Card Component ─── */
function ScreenshotCard({ imageUrl, darkMode }) {
  const [fullscreen, setFullscreen] = useState(false);

  if (!imageUrl) return null;

  return (
    <div className={`mt-3 rounded-xl border p-3.5 text-left ${
      darkMode ? 'bg-[#0c0d18] border-white/[0.08]' : 'bg-white border-cyan-100 shadow-sm'
    }`}>
      {/* Header */}
      <div className={`flex items-center justify-between gap-3 mb-3 pb-3 border-b ${
        darkMode ? 'border-white/[0.06]' : 'border-cyan-50'
      }`}>
        <div className="flex items-center gap-2.5 min-w-0">
          <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${
            darkMode ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/20' : 'bg-cyan-50 text-cyan-600 border border-cyan-200'
          }`}>
            <Sparkle weight="fill" className="w-4 h-4" />
          </div>
          <div className="min-w-0">
            <p className={`text-[12px] font-semibold ${
              darkMode ? 'text-slate-200' : 'text-slate-700'
            }`}>Screenshot Captured</p>
            <p className={`text-[10px] font-mono mt-0.5 ${
              darkMode ? 'text-slate-500' : 'text-slate-400'
            }`}>Win32 GDI · Sub-50ms</p>
          </div>
        </div>
        <span className={`shrink-0 text-[10px] px-2 py-0.5 rounded-md font-semibold font-mono ${
          darkMode
            ? 'bg-cyan-500/12 border border-cyan-500/25 text-cyan-400'
            : 'bg-cyan-50 border border-cyan-200 text-cyan-600'
        }`}>
          Ready
        </span>
      </div>

      {/* Thumbnail */}
      <div
        className={`relative mb-3 rounded-lg overflow-hidden border group cursor-pointer ${
          darkMode ? 'border-white/[0.07]' : 'border-slate-100'
        }`}
        onClick={() => setFullscreen(true)}
      >
        <img
          src={imageUrl}
          alt="Desktop Screenshot"
          className="w-full max-h-52 object-cover object-top transition-transform duration-300 group-hover:scale-[1.02]"
        />
        <div className="absolute inset-0 bg-black/30 opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex items-center justify-center">
          <span className="text-[11px] font-semibold text-white bg-black/50 px-3 py-1.5 rounded-lg">View full size</span>
        </div>
      </div>

      {/* Actions */}
      <div className="flex flex-wrap items-center gap-1.5">
        <a
          href={imageUrl}
          target="_blank"
          rel="noopener noreferrer"
          className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[11px] font-medium border transition-all duration-150 cursor-pointer active:scale-95 ${
            darkMode
              ? 'bg-white/[0.04] hover:bg-cyan-500/10 border-white/[0.07] hover:border-cyan-500/25 text-slate-400 hover:text-cyan-400'
              : 'bg-white hover:bg-cyan-50 border-cyan-100 text-slate-500 hover:text-cyan-600'
          }`}
        >
          <span>View Full ↗</span>
        </a>
        <a
          href={imageUrl}
          download="desktop_screenshot.png"
          className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[11px] font-medium border transition-all duration-150 cursor-pointer active:scale-95 ${
            darkMode
              ? 'bg-white/[0.04] hover:bg-violet-500/10 border-white/[0.07] hover:border-violet-500/25 text-slate-400 hover:text-violet-400'
              : 'bg-white hover:bg-violet-50 border-violet-100 text-slate-500 hover:text-violet-600'
          }`}
        >
          <DownloadSimple weight="bold" className="w-3 h-3" />
          <span>Save</span>
        </a>
      </div>

      {/* Fullscreen Lightbox */}
      {fullscreen && (
        <div
          className="fixed inset-0 z-50 bg-black/92 backdrop-blur-xl flex items-center justify-center p-4 cursor-pointer"
          onClick={() => setFullscreen(false)}
        >
          <div className="relative max-w-5xl max-h-[92vh]" onClick={(e) => e.stopPropagation()}>
            <img
              src={imageUrl}
              alt="Desktop Screenshot Full"
              className="max-w-full max-h-[88vh] rounded-xl shadow-2xl border border-white/[0.1] object-contain"
            />
            <button
              onClick={() => setFullscreen(false)}
              className="absolute top-3 right-3 w-8 h-8 rounded-lg bg-black/70 hover:bg-black text-white border border-white/20 flex items-center justify-center cursor-pointer transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

/* ─── EdgeVoice Realtime Voice Mode Modal ─── */
function EdgeVoiceLiveModal({
  isOpen,
  onClose,
  voiceState,
  liveTranscript,
  botResponseText,
  onInterrupt,
  onToggleMic,
  isMicMuted,
  onDoneSpeaking,
  latencyMs,
  micVolume = 0,
}) {
  if (!isOpen) return null;

  return (
    <div className="edgevoice-overlay">
      {/* Top Header */}
      <div className="flex items-center justify-between w-full max-w-3xl px-6 py-5 mx-auto">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-violet-600 to-cyan-500 flex items-center justify-center shadow-lg shadow-violet-500/25 shrink-0">
            <Headset weight="fill" className="w-4.5 h-4.5 text-white" />
          </div>
          <div>
            <h3 className="font-semibold text-[14px] text-white tracking-tight">EdgeVoice Live</h3>
            <p className="text-[10px] text-slate-500 font-mono mt-0.5">100% Offline · Zero Cloud</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {latencyMs > 0 && (
            <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-white/[0.05] border border-white/[0.07] text-[11px] font-mono text-emerald-400">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              {latencyMs}ms
            </div>
          )}
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-lg flex items-center justify-center bg-white/[0.05] hover:bg-white/[0.1] border border-white/[0.08] text-slate-400 hover:text-white transition-all duration-150 cursor-pointer"
            title="Exit Voice Mode"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Main Center Stage */}
      <div className="flex-1 flex flex-col items-center justify-center px-6 max-w-2xl mx-auto w-full text-center">
        {/* Status Badge */}
        <div className="flex items-center justify-center mb-6">
          {isMicMuted ? (
            <span className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-rose-500/15 border border-rose-500/30 text-rose-300 text-[11px] font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-rose-500" />
              Microphone Muted
            </span>
          ) : voiceState === 'listening' ? (
            micVolume > 4 ? (
              <span className="inline-flex items-center gap-2 px-4 py-1.5 rounded-lg bg-cyan-500/15 border border-cyan-400/35 text-cyan-200 text-[11px] font-mono font-semibold">
                <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
                Voice Detected · {micVolume}%
              </span>
            ) : (
              <span className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-white/[0.04] border border-white/[0.08] text-slate-400 text-[11px] font-mono">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                Listening · Speak now
              </span>
            )
          ) : voiceState === 'thinking' ? (
            <span className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-amber-500/12 border border-amber-500/25 text-amber-300 text-[11px] font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-spin" />
              AI Processing
            </span>
          ) : voiceState === 'speaking' ? (
            <span className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-violet-500/15 border border-violet-500/25 text-violet-300 text-[11px] font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-pulse" />
              EdgeVoice Speaking
            </span>
          ) : null}
        </div>

        {/* Animated Orb */}
        <div className="relative mb-8 flex items-center justify-center">
          <div className={`absolute w-64 h-64 rounded-full transition-opacity duration-700 pointer-events-none ${
            voiceState === 'listening' ? 'bg-cyan-500/12 blur-3xl opacity-100' :
            voiceState === 'speaking' ? 'bg-violet-500/18 blur-3xl opacity-100' :
            voiceState === 'thinking' ? 'bg-fuchsia-500/14 blur-3xl opacity-100' :
            'opacity-0'
          }`} />
          <div
            className={`w-36 h-36 sm:w-44 sm:h-44 rounded-full flex items-center justify-center shadow-2xl transition-all duration-300 cursor-pointer ${
              isMicMuted ? 'orb-idle' :
              voiceState === 'listening' ? 'orb-listening' :
              voiceState === 'thinking' ? 'orb-thinking' :
              voiceState === 'speaking' ? 'orb-speaking' :
              'orb-idle'
            }`}
            style={{
              transform: voiceState === 'listening' && micVolume > 3
                ? `scale(${1 + Math.min(0.22, (micVolume / 100) * 0.30)})`
                : undefined,
              boxShadow: voiceState === 'listening' && micVolume > 5
                ? `0 0 ${18 + micVolume * 0.7}px rgba(6, 182, 212, ${0.35 + (micVolume / 100) * 0.45})`
                : undefined,
            }}
            onClick={voiceState === 'speaking' ? onInterrupt : onDoneSpeaking}
            title={voiceState === 'speaking' ? 'Click to interrupt' : 'Click when done speaking'}
          >
            <div className="w-14 h-14 rounded-full bg-black/35 backdrop-blur-md flex items-center justify-center border border-white/15 shadow-inner">
              {isMicMuted ? (
                <MicrophoneSlash className="w-6 h-6 text-slate-400" />
              ) : voiceState === 'listening' ? (
                <Microphone weight="fill" className={`w-6 h-6 ${micVolume > 5 ? 'text-cyan-200 animate-bounce' : 'text-cyan-400 animate-pulse'}`} />
              ) : voiceState === 'thinking' ? (
                <Sparkle weight="fill" className="w-6 h-6 text-amber-300 animate-spin" />
              ) : voiceState === 'speaking' ? (
                <Waveform weight="bold" className="w-6 h-6 text-white animate-pulse" />
              ) : (
                <Microphone className="w-6 h-6 text-slate-500" />
              )}
            </div>
          </div>
        </div>

        {/* State Label */}
        <div className="mb-5">
          <p className="text-[15px] font-semibold tracking-tight text-white">
            {isMicMuted
              ? 'Microphone Paused'
              : voiceState === 'listening'
              ? (micVolume > 4 ? 'Voice Detected!' : 'Listening…')
              : voiceState === 'thinking'
              ? 'Generating response…'
              : voiceState === 'speaking'
              ? 'EdgeVoice is speaking'
              : 'Ready'}
          </p>
          <p className="text-[11px] text-slate-500 font-mono mt-1">
            {voiceState === 'listening'
              ? 'Speak naturally · pause or tap Done when finished'
              : voiceState === 'speaking'
              ? 'Tap Interrupt to take over'
              : voiceState === 'thinking'
              ? 'Qwen 1.5B + Piper TTS'
              : 'Hands-free conversational AI'}
          </p>
        </div>

        {/* Live Transcript Card */}
        <div className="w-full max-h-36 overflow-y-auto px-4 py-3.5 rounded-xl bg-white/[0.035] border border-white/[0.07] text-left">
          {voiceState === 'speaking' && botResponseText ? (
            <div>
              <p className="text-[10px] font-mono text-cyan-400 uppercase tracking-widest mb-1.5 font-semibold flex items-center gap-1.5">
                <Robot className="w-3 h-3" /> Assistant
              </p>
              <p className="text-[13px] text-slate-100 leading-relaxed">
                {botResponseText}
              </p>
            </div>
          ) : (
            <div>
              <p className="text-[10px] font-mono text-violet-400 uppercase tracking-widest mb-1.5 font-semibold flex items-center gap-1.5">
                <User className="w-3 h-3" /> You
              </p>
              <p className="text-[13px] text-slate-300 leading-relaxed">
                {liveTranscript || (voiceState === 'listening' ? 'Start speaking anytime…' : 'Awaiting…')}
              </p>
            </div>
          )}
        </div>

        {/* Equalizer */}
        <div className="flex items-center justify-center gap-1 mt-4 h-8 px-4 py-1 rounded-xl bg-black/30 border border-white/[0.07]">
          {[0.4, 0.7, 1.0, 0.8, 1.3, 0.9, 1.4, 1.1, 1.3, 0.8, 1.0, 0.6, 0.9, 0.5].map((multiplier, i) => {
            const barHeight = voiceState === 'speaking'
              ? 8 + (i % 4) * 5
              : voiceState === 'listening'
              ? Math.max(3, Math.min(26, Math.round(3 + (micVolume / 100) * 23 * multiplier)))
              : 3;
            return (
              <div
                key={i}
                className={`w-[3px] rounded-full transition-all duration-75 ${
                  voiceState === 'speaking'
                    ? 'bg-gradient-to-t from-violet-500 to-cyan-300 animate-pulse'
                    : micVolume > 4
                    ? 'bg-gradient-to-t from-cyan-400 to-emerald-400'
                    : voiceState === 'listening'
                    ? 'bg-cyan-500/25'
                    : 'bg-white/10'
                }`}
                style={{ height: `${barHeight}px` }}
              />
            );
          })}
        </div>
      </div>

      {/* Bottom Controls */}
      <div className="w-full max-w-sm px-6 py-6 mx-auto flex items-center justify-center gap-3">
        <button
          onClick={onToggleMic}
          className={`flex items-center justify-center w-11 h-11 rounded-xl border transition-all duration-150 cursor-pointer ${
            isMicMuted
              ? 'bg-rose-500/18 border-rose-500/35 text-rose-300'
              : 'bg-white/[0.06] hover:bg-white/[0.12] border-white/[0.09] text-slate-300 hover:text-white'
          }`}
          title={isMicMuted ? 'Unmute' : 'Mute'}
        >
          {isMicMuted ? <MicrophoneSlash weight="fill" className="w-4.5 h-4.5" /> : <Microphone weight="fill" className="w-4.5 h-4.5" />}
        </button>

        {voiceState === 'listening' && (
          <button
            onClick={onDoneSpeaking}
            className="flex-1 flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-violet-600 to-cyan-600 hover:brightness-110 text-white font-semibold text-[13px] shadow-lg shadow-violet-600/25 transition-all duration-150 cursor-pointer active:scale-[0.97]"
          >
            <PaperPlaneRight weight="fill" className="w-4 h-4" />
            Done Speaking
          </button>
        )}

        {voiceState === 'speaking' && (
          <button
            onClick={onInterrupt}
            className="flex-1 flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-rose-600 to-red-600 hover:brightness-110 text-white font-semibold text-[13px] shadow-lg shadow-rose-600/25 transition-all duration-150 cursor-pointer active:scale-[0.97]"
          >
            <Stop weight="fill" className="w-4 h-4" />
            Interrupt
          </button>
        )}

        {voiceState === 'thinking' && (
          <div className="flex-1 flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl bg-white/[0.06] border border-white/[0.08] text-slate-400 font-mono text-[12px]">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping" />
            Processing…
          </div>
        )}

        <button
          onClick={onClose}
          className="flex items-center justify-center w-11 h-11 rounded-xl bg-white/[0.06] hover:bg-rose-500/15 border border-white/[0.09] hover:border-rose-500/25 text-slate-400 hover:text-rose-300 transition-all duration-150 cursor-pointer"
          title="Exit voice mode"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}

/* ─── Main App ─── */
export default function App() {
  const [darkMode, setDarkMode] = useState(true);
  const [isListening, setIsListening] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [inputText, setInputText] = useState('');
  const [imagePreview, setImagePreview] = useState(null);
  const [imageFile, setImageFile] = useState(null);       // raw File for upload
  const [selectedFile, setSelectedFile] = useState(null); // File object for PDF/Image chip
  const [imageStage, setImageStage] = useState('idle');   // idle|uploading|describing|speaking
  const [showSettings, setShowSettings] = useState(false);
  const [activeTab, setActiveTab] = useState('chat');

  // EdgeVoice Live Autonomous Voice Mode states
  const [isVoiceModalOpen, setIsVoiceModalOpen] = useState(false);
  const [voiceState, setVoiceState] = useState('idle'); // 'listening' | 'thinking' | 'speaking' | 'idle'
  const [liveTranscript, setLiveTranscript] = useState('');
  const [botVoiceReply, setBotVoiceReply] = useState('');
  const [isMicMuted, setIsMicMuted] = useState(false);
  const [micVolume, setMicVolume] = useState(0);

  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  const wsRef = useRef(null);                             // persistent WebSocket
  const audioContextRef = useRef(null);
  const processorRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);
  const recognitionRef = useRef(null);
  const speechTextRef = useRef('');
  const activeAudioRef = useRef(null);
  const isVoiceModalOpenRef = useRef(false);
  const isMicMutedRef = useRef(false);
  const voiceStateRef = useRef(voiceState);
  const silenceTimerRef = useRef(null);
  const hasUserSpokenRef = useRef(false);
  const triggerDoneSpeakingVoiceRef = useRef(null);

  useEffect(() => { isVoiceModalOpenRef.current = isVoiceModalOpen; }, [isVoiceModalOpen]);
  useEffect(() => { isMicMutedRef.current = isMicMuted; }, [isMicMuted]);
  useEffect(() => { voiceStateRef.current = voiceState; }, [voiceState]);

  // Live system state
  const [systemState, setSystemState] = useState({
    ram_mb: 24.3, ram_cap: 3072, cpu_pct: 14.2,
    latency_ms: 106, tier: 'Tier 1 (High Quality)', status: 'Ready',
  });

  // Conversation history
  const [sessions, setSessions] = useState([
    {
      id: 'session-1',
      title: '48 × 12 & Gravity',
      date: 'Today',
      messages: [
        { id: '1', role: 'user',      text: 'What is 48 multiplied by 12?',         time: '5m ago' },
        { id: '2', role: 'assistant', text: 'The answer is 576.',                    route: 'Fast-Path', latency: 106, breakdown: 'ASR 54ms • Route 2ms • TTS 22ms' },
        { id: '3', role: 'user',      text: 'Explain gravity simply in one sentence.',time: '4m ago' },
        { id: '4', role: 'assistant', text: 'Gravity is the invisible force that pulls objects toward each other, giving weight to physical matter.', route: 'Qwen 1.5B', latency: 580, breakdown: 'ASR 68ms • LLM 410ms • TTS 70ms' },
      ],
    },
    { id: 'session-2', title: 'First Aid — Burn care', date: 'Yesterday', messages: [
      { id: '5', role: 'user', text: 'How to treat a burn?', time: 'Yesterday' },
      { id: '6', role: 'assistant', text: 'Cool the burn under cold running water for 10–20 minutes. Cover with a sterile dressing. Do not apply butter or ice.', route: 'Qwen 1.5B', latency: 612 },
    ]},
  ]);

  const [activeSessionId, setActiveSessionId] = useState('session-1');
  const activeSession = sessions.find((s) => s.id === activeSessionId);

  // Dark/Light toggle
  useEffect(() => {
    document.body.classList.toggle('light', !darkMode);
  }, [darkMode]);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activeSession?.messages, isProcessing]);

  // Backend status polling
  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch('/api/status');
        if (res.ok) {
          const d = await res.json();
          setSystemState((prev) => ({
            ...prev,
            ram_mb:   d.live_metrics?.process_ram_mb   || prev.ram_mb,
            cpu_pct:  d.live_metrics?.system_cpu_pct   || prev.cpu_pct,
            tier:     d.governor_tier                  || prev.tier,
          }));
        }
      } catch { /* offline fallback — do nothing */ }
    };
    poll();
    const t = setInterval(poll, 2500);
    return () => clearInterval(t);
  }, []);

  // Persistent WebSocket for image progress events
  useEffect(() => {
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const wsUrl = `${proto}://${window.location.host}/ws/telemetry`;
    let ws;
    let alive = true;

    const connect = () => {
      ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === 'image_progress') {
            const stageMap = {
              uploading:  'uploading',
              describing: 'describing',
              speaking:   'speaking',
            };
            setImageStage(stageMap[msg.stage] || 'describing');
          } else if (msg.type === 'image_result') {
            const res = msg.result || {};
            setImageStage('idle');
            setIsProcessing(false);
            const isPdf = res.file_type === 'pdf';
            const description = res.text || (res.success === false ? res.error : 'No description returned.');
            setSessions((prev) => {
              return prev.map((s) => {
                if (s.id !== activeSessionIdRef.current) return s;
                // Avoid duplicate if HTTP response already added this result
                const alreadyExists = s.messages.some((m) => m.wordUrl && m.wordUrl === res.word_url);
                if (alreadyExists) return s;
                return {
                  ...s,
                  messages: [
                    ...s.messages,
                    {
                      id: Date.now().toString(),
                      role: 'assistant',
                      text: description,
                      cleanText: res.clean_text,
                      wordUrl: res.word_url,
                      wordFilename: res.word_filename,
                      fileType: res.file_type || (isPdf ? 'pdf' : 'image'),
                      pageCount: res.page_count,
                      imageUrl: res.image_url,
                      filename: res.filename,
                      route: isPdf ? 'PDF-to-Word' : 'RapidOCR',
                      latency: Math.round(res.total_ms || 0),
                      breakdown: isPdf
                        ? `pypdf extraction • ${res.page_count || 1} pages converted to .docx`
                        : `RapidOCR ${Math.round(res.vision_ms || 0)}ms • RAM peak ${Math.round(res.peak_ram_mb || 0)}MB`,
                    },
                  ],
                };
              });
            });
          }
        } catch { /* ignore parse errors */ }
      };
      ws.onclose = () => {
        if (alive) setTimeout(connect, 2000);   // auto-reconnect
      };
    };

    connect();
    return () => {
      alive = false;
      ws?.close();
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Ref to always have latest activeSessionId inside WS callback
  const activeSessionIdRef = useRef(activeSessionId);
  useEffect(() => { activeSessionIdRef.current = activeSessionId; }, [activeSessionId]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 160) + 'px';
    }
  }, [inputText]);

  /* ── Helper to play base64 audio (Crisp, Single-Stream, Zero Echo) ── */
  const playAudioBase64 = useCallback((b64, onEnded) => {
    if (!b64) {
      if (onEnded) onEnded();
      return;
    }
    try {
      if (activeAudioRef.current) {
        activeAudioRef.current.pause();
        activeAudioRef.current.currentTime = 0;
        activeAudioRef.current.src = '';
        activeAudioRef.current = null;
      }
      const audio = new Audio(`data:audio/wav;base64,${b64}`);
      activeAudioRef.current = audio;
      audio.onended = () => {
        activeAudioRef.current = null;
        if (onEnded) onEnded();
      };
      audio.onerror = (e) => {
        console.warn('Audio error:', e);
        activeAudioRef.current = null;
        if (onEnded) onEnded();
      };
      audio.play().catch((err) => {
        console.warn('Audio playback prevented or interrupted:', err);
        activeAudioRef.current = null;
        if (onEnded) onEnded();
      });
    } catch (e) {
      if (onEnded) onEnded();
    }
  }, []);

  /* ── Send query ── */
  const sendMessage = useCallback(async (text, img = null, rawFile = null) => {
    if ((!text.trim() && !img && !rawFile) || isProcessing) return;
    setIsProcessing(true);
    setInputText('');
    setImagePreview(null);
    setImageFile(null);
    setSelectedFile(null);

    const isPdfUpload = rawFile && (rawFile.name?.toLowerCase().endsWith('.pdf') || rawFile.type === 'application/pdf');

    const userMsg = {
      id: Date.now().toString(),
      role: 'user',
      text: text.trim() || (isPdfUpload ? `Uploaded PDF: ${rawFile.name}` : img ? '[Image uploaded]' : ''),
      image: isPdfUpload ? null : img,
      time: 'Just now',
    };

    setSessions((prev) => prev.map((s) =>
      s.id === activeSessionId
        ? { ...s, messages: [...s.messages, userMsg], title: text.slice(0, 30) || (rawFile ? rawFile.name : s.title) }
        : s
    ));

    // ── File upload path (PDF or Image): POST to /api/upload ──
    if (rawFile) {
      try {
        setImageStage('uploading');
        const form = new FormData();
        form.append('file', rawFile);
        const res = await fetch('/api/upload', { method: 'POST', body: form });
        const data = await res.json();
        setImageStage('idle');
        setIsProcessing(false);
        if (res.ok && data.status === 'ok') {
          const r = data.result || {};
          const isPdf = r.file_type === 'pdf' || isPdfUpload;
          const botMsg = {
            id: (Date.now() + 1).toString(),
            role: 'assistant',
            text: r.text || (isPdf ? 'PDF converted to Word (.docx) successfully.' : 'Extracted text from image.'),
            cleanText: r.clean_text,
            wordUrl: r.word_url,
            wordFilename: r.word_filename,
            fileType: r.file_type || (isPdf ? 'pdf' : 'image'),
            pageCount: r.page_count,
            imageUrl: r.image_url,
            filename: rawFile.name,
            route: isPdf ? 'PDF-to-Word' : 'RapidOCR',
            latency: Math.round(r.total_ms || 0),
            breakdown: isPdf 
              ? `pypdf extraction • ${r.page_count || 1} pages converted to Microsoft Word (.docx)`
              : `RapidOCR ${Math.round(r.vision_ms || 0)}ms • RAM peak ${Math.round(r.peak_ram_mb || 0)}MB`,
          };
          setSessions((prev) => prev.map((s) =>
            s.id === activeSessionId ? { ...s, messages: [...s.messages, botMsg] } : s
          ));
        } else {
          const errMsg = {
            id: (Date.now() + 1).toString(),
            role: 'assistant',
            text: data?.error || `Upload failed (${res.status}).`,
            route: 'Error',
          };
          setSessions((prev) => prev.map((s) =>
            s.id === activeSessionId ? { ...s, messages: [...s.messages, errMsg] } : s
          ));
        }
      } catch (err) {
        setImageStage('idle');
        setIsProcessing(false);
        const errMsg = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          text: 'Upload failed. Is the server running?',
          route: 'Error',
        };
        setSessions((prev) => prev.map((s) =>
          s.id === activeSessionId ? { ...s, messages: [...s.messages, errMsg] } : s
        ));
      }
      return;
    }

    // ── Text path: POST to /api/simulate ──
    try {
      const res = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      const ev = data.event || {};
      const botReply = ev.bot_reply || data.reply || 'Completed successfully.';
      const audioB64 = ev.audio_base64;

      const botMsg = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        text: botReply,
        audioBase64: audioB64,
        screenshotUrl: ev.screenshot_url || (botReply.includes('data/screenshots/') ? ('/data/screenshots/' + (botReply.match(/screenshot_[a-zA-Z0-9_\-]+\.png/) || [])[0]) : null),
        route: ev.route || (ev.is_fast_path ? 'Fast-Path' : 'Qwen 1.5B'),
        latency: ev.total_ms || 108,
        breakdown: `Route ${ev.router_ms || 0}ms • LLM ${ev.llm_ms || 0}ms • TTS ${ev.tts_ms || 0}ms`,
      };
      setSessions((prev) => prev.map((s) =>
        s.id === activeSessionId ? { ...s, messages: [...s.messages, botMsg] } : s
      ));
      setSystemState((prev) => ({ ...prev, latency_ms: ev.total_ms || prev.latency_ms }));

      // Auto-play voice response (single stream, browser only, zero echo)
      if (audioB64) {
        playAudioBase64(audioB64);
      }
    } catch {
      await new Promise((r) => setTimeout(r, 600));
      const fallback = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        text: `Offline response for "${text}". Zero cloud calls.`,
        route: 'Fast-Path',
        latency: 112,
        breakdown: 'ASR 56ms • Route 2ms • TTS 24ms',
      };
      setSessions((prev) => prev.map((s) =>
        s.id === activeSessionId ? { ...s, messages: [...s.messages, fallback] } : s
      ));
    } finally {
      setIsProcessing(false);
    }
  }, [activeSessionId, isProcessing, playAudioBase64]);

  /* ── Replay message audio ── */
  const playMessageAudio = useCallback(async (msg) => {
    if (msg.audioBase64) {
      playAudioBase64(msg.audioBase64);
      return;
    }
    try {
      const res = await fetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: msg.text }),
      });
      const data = await res.json();
      if (data.status === 'ok' && data.audio_base64) {
        msg.audioBase64 = data.audio_base64;
        playAudioBase64(data.audio_base64);
      }
    } catch (e) {
      console.warn('TTS playback error:', e);
    }
  }, [playAudioBase64]);

  /* ── Real Microphone Recording & Offline ASR Transcription ── */
  const stopRecording = useCallback(async () => {
    setIsListening(false);
    setMicVolume(0);

    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    hasUserSpokenRef.current = false;

    if (recognitionRef.current) {
      try { recognitionRef.current.stop(); } catch (e) {}
      recognitionRef.current = null;
    }

    if (processorRef.current) {
      try { processorRef.current.disconnect(); } catch (e) {}
      processorRef.current = null;
    }
    if (streamRef.current) {
      try { streamRef.current.getTracks().forEach((t) => t.stop()); } catch (e) {}
      streamRef.current = null;
    }
    const sampleRate = audioContextRef.current ? audioContextRef.current.sampleRate : 16000;
    if (audioContextRef.current) {
      try { audioContextRef.current.close(); } catch (e) {}
      audioContextRef.current = null;
    }

    const liveText = speechTextRef.current.trim() || inputText.trim();
    speechTextRef.current = '';

    if (liveText) {
      sendMessage(liveText);
      return;
    }

    const chunks = chunksRef.current;
    chunksRef.current = [];
    if (!chunks.length) return;

    const totalLen = chunks.reduce((acc, c) => acc + c.length, 0);
    if (totalLen < 1600) return;

    const samples = new Float32Array(totalLen);
    let off = 0;
    for (const c of chunks) {
      samples.set(c, off);
      off += c.length;
    }

    setIsTranscribing(true);
    try {
      const wavBlob = encodeWAV(samples, sampleRate);
      const formData = new FormData();
      formData.append('file', wavBlob, 'recording.wav');
      const res = await fetch('/api/transcribe', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (data.status === 'ok' && data.text) {
        sendMessage(data.text);
      }
    } catch (err) {
      console.warn('Transcription error:', err);
    } finally {
      setIsTranscribing(false);
    }
  }, [sendMessage, inputText]);

  const startRecording = useCallback(async () => {
    try {
      speechTextRef.current = '';
      chunksRef.current = [];
      hasUserSpokenRef.current = false;
      if (silenceTimerRef.current) {
        clearTimeout(silenceTimerRef.current);
        silenceTimerRef.current = null;
      }

      // 1. Web Speech recognition for real-time live preview
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecognition) {
        try {
          const recog = new SpeechRecognition();
          recog.lang = 'en-US';
          recog.interimResults = true;
          recog.continuous = true;
          recog.onresult = (e) => {
            let finalStr = '';
            let interimStr = '';
            for (let i = 0; i < e.results.length; i++) {
              const res = e.results[i];
              if (res && res[0]) {
                const text = res[0].transcript || '';
                if (res.isFinal) {
                  finalStr += (finalStr ? ' ' : '') + text.trim();
                } else {
                  interimStr += (interimStr ? ' ' : '') + text.trim();
                }
              }
            }
            if (finalStr) {
              speechTextRef.current = finalStr;
            }
            const fullTranscript = (finalStr + (interimStr ? ' ' + interimStr : '')).trim();
            if (fullTranscript) {
              setInputText(fullTranscript);
              setLiveTranscript(fullTranscript);
              hasUserSpokenRef.current = true;

              // In Live Voice Mode, automatically detect pause in speech to auto-send
              if (isVoiceModalOpenRef.current && voiceStateRef.current === 'listening') {
                if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
                silenceTimerRef.current = setTimeout(() => {
                  silenceTimerRef.current = null;
                  hasUserSpokenRef.current = false;
                  if (triggerDoneSpeakingVoiceRef.current) {
                    triggerDoneSpeakingVoiceRef.current();
                  }
                }, 1800);
              }
            }
          };
          recog.onerror = () => {};
          recog.onend = () => {};
          recog.start();
          recognitionRef.current = recog;
        } catch (e) {}
      }

      // 2. Capture microphone audio stream with echo cancellation
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      streamRef.current = stream;

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const audioCtx = new AudioCtx();
      audioContextRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        chunksRef.current.push(new Float32Array(input));

        // Real-time audio volume calculation
        let sum = 0;
        for (let i = 0; i < input.length; i++) {
          sum += input[i] * input[i];
        }
        const rms = Math.sqrt(sum / input.length);
        const volPct = Math.min(100, Math.round(rms * 450));
        setMicVolume(volPct);

        // In Live Voice Mode, monitor audio level for silence detection
        if (isVoiceModalOpenRef.current && voiceStateRef.current === 'listening') {
          if (rms > 0.02) {
            hasUserSpokenRef.current = true;
            if (silenceTimerRef.current) {
              clearTimeout(silenceTimerRef.current);
              silenceTimerRef.current = null;
            }
          } else if (hasUserSpokenRef.current) {
            if (!silenceTimerRef.current) {
              silenceTimerRef.current = setTimeout(() => {
                silenceTimerRef.current = null;
                hasUserSpokenRef.current = false;
                if (triggerDoneSpeakingVoiceRef.current) {
                  triggerDoneSpeakingVoiceRef.current();
                }
              }, 1800);
            }
          }
        }
      };

      // Mute gain node: ensures microphone audio is NEVER fed back into laptop speakers!
      const zeroGain = audioCtx.createGain();
      zeroGain.gain.value = 0;
      source.connect(processor);
      processor.connect(zeroGain);
      zeroGain.connect(audioCtx.destination);

      setIsListening(true);
    } catch (err) {
      console.error('Microphone access error:', err);
      setIsListening(false);
      alert('Microphone access could not be opened. Please allow microphone permissions in your browser.');
    }
  }, []);

  /* ── Voice toggle ── */
  const toggleVoice = () => {
    if (isListening) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  /* ── EdgeVoice Live Voice Mode Controls ── */
  const openVoiceMode = useCallback(() => {
    setIsVoiceModalOpen(true);
    setVoiceState('listening');
    setLiveTranscript('');
    setBotVoiceReply('');
    setIsMicMuted(false);
    startRecording();
  }, [startRecording]);

  const closeVoiceMode = useCallback(() => {
    setIsVoiceModalOpen(false);
    setVoiceState('idle');
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    hasUserSpokenRef.current = false;
    stopRecording();
    if (activeAudioRef.current) {
      activeAudioRef.current.pause();
      activeAudioRef.current.currentTime = 0;
      activeAudioRef.current.src = '';
      activeAudioRef.current = null;
    }
  }, [stopRecording]);

  const handleInterruptVoice = useCallback(() => {
    if (activeAudioRef.current) {
      activeAudioRef.current.pause();
      activeAudioRef.current.currentTime = 0;
      activeAudioRef.current.src = '';
      activeAudioRef.current = null;
    }
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    hasUserSpokenRef.current = false;
    setVoiceState('listening');
    setLiveTranscript('');
    startRecording();
  }, [startRecording]);

  const handleToggleMic = useCallback(() => {
    setIsMicMuted((prev) => {
      const next = !prev;
      if (next) {
        stopRecording();
      } else {
        if (voiceState === 'listening') {
          startRecording();
        }
      }
      return next;
    });
  }, [stopRecording, startRecording, voiceState]);

  const handleDoneSpeakingVoice = useCallback(async () => {
    if (voiceStateRef.current !== 'listening') return;
    setVoiceState('thinking');

    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    hasUserSpokenRef.current = false;

    // Stop mic recording cleanly to prevent acoustic feedback/echo
    if (recognitionRef.current) {
      try { recognitionRef.current.stop(); } catch (e) {}
      recognitionRef.current = null;
    }
    if (processorRef.current) {
      try { processorRef.current.disconnect(); } catch (e) {}
      processorRef.current = null;
    }
    if (streamRef.current) {
      try { streamRef.current.getTracks().forEach((t) => t.stop()); } catch (e) {}
      streamRef.current = null;
    }
    const sampleRate = audioContextRef.current ? audioContextRef.current.sampleRate : 16000;
    if (audioContextRef.current) {
      try { audioContextRef.current.close(); } catch (e) {}
      audioContextRef.current = null;
    }
    setIsListening(false);

    let queryText = speechTextRef.current.trim() || liveTranscript.trim();
    speechTextRef.current = '';

    // If Web Speech didn't catch text, transcribe raw PCM audio with Faster-Whisper
    if (!queryText) {
      const chunks = chunksRef.current;
      chunksRef.current = [];
      if (chunks.length) {
        const totalLen = chunks.reduce((acc, c) => acc + c.length, 0);
        if (totalLen >= 1600) {
          const samples = new Float32Array(totalLen);
          let off = 0;
          for (const c of chunks) {
            samples.set(c, off);
            off += c.length;
          }
          try {
            const wavBlob = encodeWAV(samples, sampleRate);
            const formData = new FormData();
            formData.append('file', wavBlob, 'recording.wav');
            const res = await fetch('/api/transcribe', { method: 'POST', body: formData });
            const data = await res.json();
            if (data.status === 'ok' && data.text) {
              queryText = data.text.trim();
            }
          } catch (e) {
            console.warn('Voice transcription error:', e);
          }
        }
      }
    }

    if (!queryText) {
      setVoiceState('listening');
      startRecording();
      return;
    }

    setLiveTranscript(queryText);

    try {
      const userMsg = {
        id: Date.now().toString(),
        role: 'user',
        text: queryText,
        time: 'Just now',
      };
      setSessions((prev) => prev.map((s) =>
        s.id === activeSessionIdRef.current
          ? { ...s, messages: [...s.messages, userMsg], title: queryText.slice(0, 30) || s.title }
          : s
      ));

      const res = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: queryText }),
      });
      const data = await res.json();
      const ev = data.event || {};
      const botReply = ev.bot_reply || data.reply || 'Understood.';
      const audioB64 = ev.audio_base64;

      setBotVoiceReply(botReply);
      setVoiceState('speaking');

      const botMsg = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        text: botReply,
        audioBase64: audioB64,
        screenshotUrl: ev.screenshot_url || (botReply.includes('data/screenshots/') ? ('/data/screenshots/' + (botReply.match(/screenshot_[a-zA-Z0-9_\-]+\.png/) || [])[0]) : null),
        route: ev.route || 'Qwen 1.5B',
        latency: ev.total_ms || 108,
        breakdown: `Route ${ev.router_ms || 0}ms • LLM ${ev.llm_ms || 0}ms • TTS ${ev.tts_ms || 0}ms`,
      };
      setSessions((prev) => prev.map((s) =>
        s.id === activeSessionIdRef.current ? { ...s, messages: [...s.messages, botMsg] } : s
      ));

      if (audioB64) {
        playAudioBase64(audioB64, () => {
          // Seamless hands-free conversational loop: when speech ends, resume listening!
          if (isVoiceModalOpenRef.current && !isMicMutedRef.current) {
            setVoiceState('listening');
            setLiveTranscript('');
            startRecording();
          } else {
            setVoiceState('idle');
          }
        });
      } else {
        setTimeout(() => {
          if (isVoiceModalOpenRef.current && !isMicMutedRef.current) {
            setVoiceState('listening');
            setLiveTranscript('');
            startRecording();
          } else {
            setVoiceState('idle');
          }
        }, 1500);
      }
    } catch (e) {
      console.warn('Voice simulation error:', e);
      setVoiceState('listening');
      startRecording();
    }
  }, [liveTranscript, startRecording, playAudioBase64]);

  // Keep trigger reference updated for silence auto-send
  useEffect(() => {
    triggerDoneSpeakingVoiceRef.current = handleDoneSpeakingVoice;
  }, [handleDoneSpeakingVoice]);

  /* ── AI Solve & Summarize Handlers ── */
  const solveWithAI = useCallback((text) => {
    sendMessage(`Please solve and provide detailed step-by-step answers for all questions or tests found in this text:\n\n${text}`);
  }, [sendMessage]);

  const summarizeWithAI = useCallback((text) => {
    sendMessage(`Please summarize the following document and list the key takeaways clearly:\n\n${text}`);
  }, [sendMessage]);

  /* ── New session ── */
  const newSession = () => {
    const id = `session-${Date.now()}`;
    setSessions((prev) => [{ id, title: 'New Chat', date: 'Just now', messages: [] }, ...prev]);
    setActiveSessionId(id);
  };

  /* ── Delete session ── */
  const deleteSession = (id) => {
    setSessions((prev) => prev.filter((s) => s.id !== id));
    if (activeSessionId === id) {
      const remaining = sessions.filter((s) => s.id !== id);
      setActiveSessionId(remaining[0]?.id || null);
    }
  };

  /* ── File upload: preview locally + keep raw File for upload (Images & PDFs) ── */
  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setImageFile(file);
    setSelectedFile(file);
    const isPdf = file.name.toLowerCase().endsWith('.pdf') || file.type === 'application/pdf';
    if (isPdf) {
      setImagePreview(null);
    } else {
      const reader = new FileReader();
      reader.onload = (ev) => setImagePreview(ev.target.result);
      reader.readAsDataURL(file);
    }
    // Reset input so the same file can be re-selected after clearing
    e.target.value = '';
  };
  const handleImageUpload = handleFileUpload;

  /* ── Key handler ── */
  const onKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(inputText, imagePreview, imageFile);
    }
  };

  /* ── Image-stage label ── */
  const stageLabelMap = {
    idle: 'Processing…',
    uploading: 'Uploading image…',
    describing: 'Describing image (offline AI)…',
    speaking: 'Generating speech…',
  };

  return (
    <div className={`min-h-[100dvh] flex flex-col font-sans ${darkMode ? 'bg-[#09090f] text-slate-100' : 'bg-[#f8f8fc] text-[#16142e]'}`}>

      {/* ── Settings Modal ── */}
      {showSettings && (
        <SettingsModal
          onClose={() => setShowSettings(false)}
          darkMode={darkMode}
          onToggleDark={() => setDarkMode((d) => !d)}
        />
      )}

      {/* ── EdgeVoice Autonomous Real-Time Voice Mode Modal ── */}
      <EdgeVoiceLiveModal
        isOpen={isVoiceModalOpen}
        onClose={closeVoiceMode}
        voiceState={voiceState}
        liveTranscript={liveTranscript}
        botResponseText={botVoiceReply}
        onInterrupt={handleInterruptVoice}
        onToggleMic={handleToggleMic}
        isMicMuted={isMicMuted}
        onDoneSpeaking={handleDoneSpeakingVoice}
        latencyMs={systemState.latency_ms}
        micVolume={micVolume}
      />

      {/* ── LAYOUT: Sidebar + Main ── */}
      <div className="flex flex-1 overflow-hidden h-[100dvh]">

        {/* ═══════════════════════════════════════
            LEFT SIDEBAR
        ═══════════════════════════════════════ */}
        <aside className={`hidden md:flex flex-col w-58 shrink-0 border-r p-2.5 gap-2 ${
          darkMode
            ? 'border-white/[0.05] bg-[#060610]'
            : 'border-violet-100 bg-[#f3f1fb]'
        }`}>

          {/* Brand */}
          <div className="flex items-center gap-2.5 px-2 pt-3 pb-2.5">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-violet-600 to-cyan-500 flex items-center justify-center shadow-sm shadow-violet-600/30 shrink-0">
              <Sparkle weight="fill" className="w-3.5 h-3.5 text-white" />
            </div>
            <div>
              <p className={`font-semibold text-[13px] tracking-tight leading-none ${darkMode ? 'text-white' : 'text-[#16142e]'}`}>EdgeVoice</p>
              <p className={`text-[9px] mt-0.5 font-mono tracking-widest uppercase ${darkMode ? 'text-slate-600' : 'text-violet-400'}`}>HNX26EPS08</p>
            </div>
          </div>

          {/* New Chat */}
          <button
            onClick={newSession}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg text-[12.5px] font-medium transition-all duration-150 ripple-btn w-full ${
              darkMode
                ? 'bg-violet-600 hover:bg-violet-500 text-white shadow-sm shadow-violet-600/20'
                : 'bg-violet-600 hover:bg-violet-700 text-white shadow-sm shadow-violet-600/20'
            }`}
          >
            <ChatCircleDots weight="fill" className="w-3.5 h-3.5 shrink-0" />
            New Chat
          </button>

          {/* Nav */}
          <div className="flex flex-col gap-0.5">
            <div
              className={`sidebar-item ${activeTab === 'chat' ? 'active' : ''}`}
              onClick={() => setActiveTab('chat')}
            >
              <Robot className="w-3.5 h-3.5 shrink-0" />
              <span>AI Chat</span>
            </div>
            <div
              className={`sidebar-item ${activeTab === 'metrics' ? 'active' : ''}`}
              onClick={() => setActiveTab('metrics')}
            >
              <Cpu className="w-3.5 h-3.5 shrink-0" />
              <span>System Metrics</span>
            </div>
          </div>

          {/* Divider */}
          <div className={`h-px my-0.5 ${darkMode ? 'bg-white/[0.04]' : 'bg-violet-100'}`} />

          {/* History */}
          <div className="flex-1 overflow-y-auto min-h-0">
            <p className={`text-[9.5px] font-semibold uppercase tracking-widest px-2.5 mb-2 ${
              darkMode ? 'text-slate-700' : 'text-slate-400'
            }`}>
              Recent
            </p>
            <div className="flex flex-col gap-0.5">
              {sessions.map((s) => (
                <div
                  key={s.id}
                  className={`sidebar-item group ${activeSessionId === s.id ? 'active' : ''}`}
                  onClick={() => { setActiveSessionId(s.id); setActiveTab('chat'); }}
                >
                  <ChatCircleDots className="w-3 h-3 shrink-0 opacity-50" />
                  <span className="flex-1 truncate text-[12px]">{s.title}</span>
                  <button
                    onClick={(e) => { e.stopPropagation(); deleteSession(s.id); }}
                    className="opacity-0 group-hover:opacity-100 p-0.5 rounded hover:text-rose-400 hover:bg-rose-500/10 transition-all duration-150"
                    title="Delete chat"
                  >
                    <Trash className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* Bottom Controls */}
          <div className={`flex flex-col gap-0.5 pt-2 border-t ${
            darkMode ? 'border-white/[0.045]' : 'border-violet-100'
          }`}>
            <div className="sidebar-item" onClick={() => setShowSettings(true)}>
              <Gear className="w-3.5 h-3.5 shrink-0" />
              <span>Settings</span>
            </div>
            <div className="sidebar-item" onClick={() => setDarkMode((d) => !d)}>
              {darkMode
                ? <SunHorizon className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                : <Moon className="w-3.5 h-3.5 text-violet-500 shrink-0" />
              }
              <span>{darkMode ? 'Light Mode' : 'Dark Mode'}</span>
            </div>
          </div>
        </aside>

        {/* ═══════════════════════════════════════
            MAIN CONTENT
        ═══════════════════════════════════════ */}
        <div className="flex flex-col flex-1 min-w-0">

          {/* ─── TOP NAVBAR ─── */}
          <header className={`shrink-0 border-b px-4 sm:px-5 h-12 flex items-center justify-between sticky top-0 z-50 ${
            darkMode
              ? 'border-white/[0.05] bg-[#09090f]/95 backdrop-blur-xl'
              : 'border-violet-100 bg-[#f8f8fc]/95 backdrop-blur-xl'
          }`}>
            <div className="flex items-center gap-2.5">
              {/* Mobile brand */}
              <div className="md:hidden w-6 h-6 rounded-lg bg-gradient-to-br from-violet-600 to-cyan-500 flex items-center justify-center shrink-0">
                <Sparkle weight="fill" className="w-3 h-3 text-white" />
              </div>
              <span className={`font-medium text-[13px] tracking-tight ${
                darkMode ? 'text-slate-200' : 'text-[#16142e]'
              }`}>
                {activeTab === 'metrics' ? 'System Metrics' : (activeSession?.title || 'EdgeVoice')}
              </span>
            </div>

            <div className="flex items-center gap-1">
              {/* Offline pill */}
              <div className={`hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] font-medium border ${
                darkMode
                  ? 'bg-white/[0.025] border-white/[0.055] text-slate-500'
                  : 'bg-emerald-50 border-emerald-100 text-emerald-600'
              }`}>
                <ShieldCheck className="w-3 h-3 text-emerald-400" />
                <span>Offline</span>
              </div>

              {/* Live status */}
              <div className={`flex items-center gap-1.5 px-2 py-1 rounded-lg text-[11px] font-medium border ${
                darkMode
                  ? 'bg-white/[0.025] border-white/[0.055] text-slate-500'
                  : 'bg-emerald-50 border-emerald-100 text-emerald-600'
              }`}>
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                <span>Live</span>
              </div>

              {/* Voice Mode */}
              <button
                onClick={openVoiceMode}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] font-medium border transition-all duration-150 cursor-pointer active:scale-95 ${
                  darkMode
                    ? 'bg-violet-500/10 hover:bg-violet-500/16 border-violet-500/20 text-violet-400 hover:text-violet-300'
                    : 'bg-violet-50 hover:bg-violet-100 border-violet-200 text-violet-600'
                }`}
                title="EdgeVoice Live Mode"
              >
                <Headset weight="fill" className="w-3 h-3" />
                <span className="hidden sm:inline">Voice</span>
              </button>

              {/* Theme toggle */}
              <button
                onClick={() => setDarkMode((d) => !d)}
                className={`w-7 h-7 rounded-lg flex items-center justify-center transition-all duration-150 ${
                  darkMode
                    ? 'hover:bg-white/[0.06] text-slate-500 hover:text-slate-300'
                    : 'hover:bg-violet-100 text-violet-400 hover:text-violet-600'
                }`}
              >
                {darkMode ? <SunHorizon className="w-3.5 h-3.5 text-amber-400" /> : <Moon className="w-3.5 h-3.5" />}
              </button>

              {/* Settings */}
              <button
                onClick={() => setShowSettings(true)}
                className={`w-7 h-7 rounded-lg flex items-center justify-center transition-all duration-150 ${
                  darkMode
                    ? 'hover:bg-white/[0.06] text-slate-500 hover:text-slate-300'
                    : 'hover:bg-violet-100 text-violet-400 hover:text-violet-600'
                }`}
              >
                <Gear className="w-3.5 h-3.5" />
              </button>
            </div>
          </header>

          {/* ─── METRICS TAB ─── */}
          {activeTab === 'metrics' && (
            <div className="flex-1 overflow-y-auto p-5 sm:p-8">
              <div className="max-w-2xl mx-auto">
                {/* Header */}
                <div className="mb-6">
                  <h1 className={`text-[15px] font-semibold tracking-tight ${
                    darkMode ? 'text-white' : 'text-[#16142e]'
                  }`}>System Metrics</h1>
                  <p className={`text-[12px] mt-0.5 ${
                    darkMode ? 'text-slate-500' : 'text-slate-500'
                  }`}>Real-time · 100% offline · zero cloud calls</p>
                </div>

                {/* Metric Cards */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
                  {[
                    {
                      label: 'Response Latency',
                      value: systemState.latency_ms,
                      unit: 'ms',
                      badge: '79% faster',
                      icon: <Clock className="w-3.5 h-3.5" />,
                      bar: '25%',
                      barColor: 'from-violet-600 to-cyan-400',
                      iconColor: 'text-violet-400',
                    },
                    {
                      label: 'RAM Working Set',
                      value: systemState.ram_mb,
                      unit: 'MB',
                      badge: 'cap 3072 MB',
                      icon: <HardDrives className="w-3.5 h-3.5" />,
                      bar: '2%',
                      barColor: 'from-emerald-600 to-emerald-400',
                      iconColor: 'text-emerald-400',
                    },
                    {
                      label: 'CPU Affinity',
                      value: systemState.cpu_pct,
                      unit: '%',
                      badge: '4 cores',
                      icon: <Cpu className="w-3.5 h-3.5" />,
                      bar: '18%',
                      barColor: 'from-violet-600 to-violet-400',
                      iconColor: 'text-violet-400',
                    },
                  ].map(({ label, value, unit, badge, icon, bar, barColor, iconColor }) => (
                    <div
                      key={label}
                      className={darkMode ? 'metric-card-dark' : 'metric-card-light'}
                    >
                      <div className="flex items-center justify-between mb-3">
                        <div className={`flex items-center gap-1.5 text-[11px] font-medium ${
                          darkMode ? 'text-slate-400' : 'text-slate-500'
                        } ${iconColor}`}>
                          {icon}
                          <span className={darkMode ? 'text-slate-400' : 'text-slate-500'}>{label}</span>
                        </div>
                        <span className={`px-1.5 py-0.5 rounded-md text-[9.5px] font-mono font-medium ${
                          darkMode
                            ? 'bg-white/[0.04] text-slate-500 border border-white/[0.06]'
                            : 'bg-violet-50 text-violet-500 border border-violet-100'
                        }`}>{badge}</span>
                      </div>
                      <div className={`text-[26px] font-bold font-mono tracking-tight mb-3 ${
                        darkMode ? 'text-white' : 'text-[#16142e]'
                      }`}>
                        {value}
                        <span className={`text-[13px] font-sans ml-1 font-normal ${
                          darkMode ? 'text-slate-500' : 'text-slate-400'
                        }`}>{unit}</span>
                      </div>
                      <div className={`w-full h-[3px] rounded-full overflow-hidden ${
                        darkMode ? 'bg-white/[0.06]' : 'bg-slate-100'
                      }`}>
                        <div className={`bg-gradient-to-r ${barColor} h-full rounded-full`} style={{ width: bar }} />
                      </div>
                    </div>
                  ))}
                </div>

                {/* Hardware info */}
                <div className={darkMode ? 'metric-card-dark' : 'metric-card-light'}>
                  <p className={`text-[10px] font-semibold uppercase tracking-widest mb-3 ${
                    darkMode ? 'text-slate-600' : 'text-slate-400'
                  }`}>Hardware Configuration</p>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                    {[
                      ['Model', 'Qwen 2.5 · 1.5B'],
                      ['Mode', '100% Offline'],
                      ['CPU', 'AMD Ryzen 7'],
                      ['OS', 'Windows 11'],
                    ].map(([k, v]) => (
                      <div key={k} className={`rounded-lg p-2.5 ${
                        darkMode ? 'bg-white/[0.03] border border-white/[0.04]' : 'bg-slate-50 border border-slate-100'
                      }`}>
                        <p className={`text-[10px] mb-1 ${
                          darkMode ? 'text-slate-600' : 'text-slate-400'
                        }`}>{k}</p>
                        <p className={`text-[11px] font-semibold font-mono ${
                          darkMode ? 'text-slate-300' : 'text-violet-700'
                        }`}>{v}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ─── CHAT TAB ─── */}
          {activeTab === 'chat' && (
            <div className="flex flex-col flex-1 min-h-0">

              {/* Messages */}
              <div className="flex-1 overflow-y-auto px-4 sm:px-6 py-5 flex flex-col gap-4">

                {/* Empty state */}
                {(!activeSession || activeSession.messages.length === 0) && (
                  <div className="flex-1 flex flex-col items-center justify-center text-center py-16 gap-6">
                    {/* Icon */}
                    <div className="relative">
                      <div className="absolute inset-0 rounded-2xl blur-2xl bg-violet-600/10 neural-dot scale-150" />
                      <div className={`relative w-14 h-14 rounded-2xl border flex items-center justify-center orb-breath ${
                        darkMode
                          ? 'bg-[#0d0a1a] border-violet-500/20'
                          : 'bg-violet-50 border-violet-200'
                      }`}>
                        <Sparkle weight="fill" className="w-6 h-6 text-violet-400" />
                      </div>
                    </div>

                    <div>
                      <h2 className={`font-semibold text-[16px] tracking-tight mb-1.5 ${
                        darkMode ? 'text-white' : 'text-[#16142e]'
                      }`}>EdgeVoice AI</h2>
                      <p className={`text-[13px] max-w-[280px] leading-relaxed ${
                        darkMode ? 'text-slate-500' : 'text-slate-500'
                      }`}>
                        100% offline · CPU only · Zero cloud<br />
                        Voice, image, text, or PDF — anything.
                      </p>
                    </div>

                    {/* Quick Prompts */}
                    <div className="flex flex-wrap justify-center gap-1.5 max-w-[360px]">
                      {[
                        { label: 'Time & Charging', query: 'time charging status' },
                        { label: 'RAM Status', query: 'ram running' },
                        { label: 'File Count', query: 'desktop la evlo files iruku' },
                        { label: 'Screenshot', query: 'screen shot edukuradhu' },
                        { label: 'Open Paint', query: 'open paint' },
                        { label: 'Close Notepad', query: 'close notepad' },
                        { label: 'Read Notes', query: 'read notes' },
                        { label: 'Treat a burn', query: 'How to treat a burn?' },
                      ].map((p, i) => (
                        <button
                          key={i}
                          onClick={() => sendMessage(p.query)}
                          disabled={isProcessing}
                          className={`px-3 py-1.5 rounded-lg text-[12px] font-medium border transition-all duration-150 active:scale-[0.97] disabled:opacity-40 ${
                            darkMode
                              ? 'bg-white/[0.03] hover:bg-white/[0.06] border-white/[0.06] hover:border-white/[0.10] text-slate-400 hover:text-slate-200'
                              : 'bg-white hover:bg-violet-50 border-slate-200 hover:border-violet-200 text-slate-600 hover:text-violet-700 shadow-sm'
                          }`}
                        >
                          {p.label}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Message bubbles */}
                {activeSession?.messages.map((m) => (
                  <div
                    key={m.id}
                    className={`flex flex-col gap-1.5 msg-enter ${
                      m.role === 'user' ? 'items-end' : 'items-start'
                    }`}
                  >
                    {/* Row header */}
                    <div className="flex items-center gap-1.5 px-0.5 w-full max-w-[680px]">
                      {m.role === 'assistant' && (
                        <div className="w-4.5 h-4.5 rounded-md bg-gradient-to-br from-violet-600 to-cyan-500 flex items-center justify-center shrink-0">
                          <Robot className="w-2.5 h-2.5 text-white" weight="fill" />
                        </div>
                      )}
                      <span className={`text-[11px] font-medium ${
                        darkMode ? 'text-slate-500' : 'text-slate-400'
                      }`}>
                        {m.role === 'user' ? 'You' : 'EdgeVoice AI'}
                      </span>
                      {m.route && (
                        <span className={`text-[10px] px-1.5 py-0.5 rounded-md font-mono ${
                          darkMode
                            ? 'bg-white/[0.04] border border-white/[0.055] text-slate-600'
                            : 'bg-violet-50 border border-violet-100 text-violet-400'
                        }`}>
                          {m.route}
                        </span>
                      )}
                      {m.latency && (
                        <span className="text-[10px] text-emerald-500/80 font-mono">{m.latency}ms</span>
                      )}
                      {m.role === 'assistant' && (
                        <button
                          onClick={() => playMessageAudio(m)}
                          title="Replay voice"
                          className={`ml-auto p-1 rounded-md transition-all duration-150 cursor-pointer ${
                            darkMode
                              ? 'text-slate-600 hover:text-slate-300 hover:bg-white/[0.05]'
                              : 'text-slate-400 hover:text-violet-500 hover:bg-violet-50'
                          }`}
                        >
                          <SpeakerHigh className="w-3 h-3" />
                        </button>
                      )}
                      {m.role === 'user' && (
                        <div className={`w-4 h-4 rounded-md flex items-center justify-center shrink-0 ml-auto ${
                          darkMode ? 'bg-slate-700/80' : 'bg-slate-200'
                        }`}>
                          <User className="w-2.5 h-2.5 text-white" weight="fill" />
                        </div>
                      )}
                    </div>

                    {/* Image preview */}
                    {m.image && (
                      <img src={m.image} alt="uploaded" className="img-preview mb-0.5" />
                    )}

                    {/* Bubble */}
                    <div className={`px-4 py-3 rounded-xl w-full max-w-[680px] text-[13px] leading-relaxed overflow-hidden ${
                      m.role === 'user'
                        ? 'bg-violet-600 text-white rounded-tr-[4px]'
                        : darkMode
                          ? 'msg-bot-dark'
                          : 'msg-bot-light'
                    }`}>
                      <FormattedMessage content={m.text} isUser={m.role === 'user'} darkMode={darkMode} />

                      {/* OCR / PDF Card */}
                      {(m.cleanText || m.wordUrl || m.fileType) && (
                        <ExtractedTextCard
                          message={m}
                          onSolveWithAI={solveWithAI}
                          onSummarizeWithAI={summarizeWithAI}
                          darkMode={darkMode}
                        />
                      )}

                      {/* Screenshot Card */}
                      {(m.screenshotUrl || (m.text && m.text.includes('data/screenshots/screenshot_'))) && (
                        <ScreenshotCard
                          imageUrl={m.screenshotUrl || ('/data/screenshots/' + (m.text.match(/screenshot_[a-zA-Z0-9_\-]+\.png/) || [])[0])}
                          darkMode={darkMode}
                        />
                      )}
                    </div>

                    {/* Breakdown */}
                    {m.breakdown && (
                      <span className={`text-[10px] font-mono px-0.5 ${
                        darkMode ? 'text-slate-700' : 'text-slate-400'
                      }`}>
                        {m.breakdown}
                      </span>
                    )}
                  </div>
                ))}

                {/* Loading state */}
                {(isProcessing || isTranscribing) && (
                  <div className="flex flex-col items-start gap-1.5 msg-enter">
                    <div className="flex items-center gap-1.5 px-0.5">
                      <div className="w-4 h-4 rounded-md bg-gradient-to-br from-violet-600 to-cyan-500 flex items-center justify-center shrink-0">
                        <Robot className="w-2.5 h-2.5 text-white" weight="fill" />
                      </div>
                      <span className={`text-[11px] font-medium ${
                        darkMode ? 'text-slate-500' : 'text-slate-400'
                      }`}>EdgeVoice AI</span>
                      {imageStage !== 'idle' && (
                        <span className={`text-[10px] px-1.5 py-0.5 rounded-md font-mono capitalize ${
                          darkMode
                            ? 'bg-amber-500/10 border border-amber-500/20 text-amber-400'
                            : 'bg-amber-50 border border-amber-100 text-amber-600'
                        }`}>
                          {imageStage}
                        </span>
                      )}
                    </div>
                    <div className={`px-4 py-3.5 rounded-xl rounded-tl-[4px] border ${
                      darkMode
                        ? 'bg-[#111120] border-white/[0.06]'
                        : 'bg-white border-slate-100 shadow-sm'
                    }`}>
                      <NeuralLoader label={
                        isTranscribing
                          ? 'Transcribing with Faster-Whisper…'
                          : (stageLabelMap[imageStage] || 'Processing…')
                      } />
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>

              {/* ─── INPUT BAR ─── */}
              <div className={`shrink-0 px-4 sm:px-5 py-3 border-t ${
                darkMode ? 'border-white/[0.05]' : 'border-violet-100/70'
              }`}>

                {/* File preview strip */}
                {(imagePreview || selectedFile) && (
                  <div className={`mb-2.5 flex items-center gap-2.5 p-2.5 rounded-xl border max-w-[680px] mx-auto ${
                    darkMode
                      ? 'bg-white/[0.025] border-white/[0.07]'
                      : 'bg-violet-50/80 border-violet-100'
                  }`}>
                    {selectedFile?.name?.toLowerCase().endsWith('.pdf') ? (
                      <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${
                        darkMode
                          ? 'bg-rose-500/12 border border-rose-500/20 text-rose-400'
                          : 'bg-rose-50 border border-rose-200 text-rose-500'
                      }`}>
                        <FilePdf weight="fill" className="w-4 h-4" />
                      </div>
                    ) : imagePreview ? (
                      <img src={imagePreview} alt="preview" className="img-preview w-8 h-8 rounded-lg object-cover shrink-0" />
                    ) : (
                      <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${
                        darkMode
                          ? 'bg-cyan-500/12 border border-cyan-500/20 text-cyan-400'
                          : 'bg-cyan-50 border border-cyan-200 text-cyan-600'
                      }`}>
                        <FileDoc weight="fill" className="w-4 h-4" />
                      </div>
                    )}
                    <div className="flex-1 min-w-0">
                      <p className={`text-[12px] font-medium truncate ${
                        darkMode ? 'text-slate-300' : 'text-slate-600'
                      }`}>
                        {selectedFile?.name || 'Selected File'}
                      </p>
                      <p className={`text-[10px] font-mono mt-0.5 ${
                        darkMode ? 'text-slate-600' : 'text-slate-400'
                      }`}>
                        {selectedFile?.name?.toLowerCase().endsWith('.pdf')
                          ? 'PDF · Auto-convert to Word (.docx)'
                          : 'Image · RapidOCR text extraction'}
                      </p>
                    </div>
                    <button
                      onClick={() => { setImagePreview(null); setSelectedFile(null); setImageFile(null); }}
                      className={`p-1 rounded-lg transition-all duration-150 cursor-pointer ${
                        darkMode
                          ? 'text-slate-600 hover:text-rose-400 hover:bg-rose-500/10'
                          : 'text-slate-400 hover:text-rose-500 hover:bg-rose-50'
                      }`}
                      title="Remove"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                )}

                {/* Mic activity indicator */}
                {isListening && (
                  <div className={`flex items-center justify-between px-3.5 py-2 mb-2.5 rounded-xl border max-w-[680px] mx-auto ${
                    darkMode
                      ? 'bg-cyan-950/40 border-cyan-500/20'
                      : 'bg-cyan-50 border-cyan-200'
                  }`}>
                    <div className="flex items-center gap-2">
                      <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                        micVolume > 4 ? 'bg-cyan-400 animate-ping' : 'bg-emerald-400 animate-pulse'
                      }`} />
                      <span className={`text-[11px] font-mono ${
                        darkMode ? 'text-cyan-400' : 'text-cyan-700'
                      }`}>
                        {micVolume > 4 ? `Hearing voice · ${micVolume}%` : 'Mic active · Speak now…'}
                      </span>
                    </div>
                    <div className="flex items-end gap-[3px] h-4">
                      {[0.4, 0.8, 1.2, 0.7, 1.4, 0.9, 1.1, 0.6, 1.3, 0.5].map((mlt, idx) => {
                        const h = Math.max(3, Math.min(14, Math.round(3 + (micVolume / 100) * 11 * mlt)));
                        return (
                          <div
                            key={idx}
                            className={`w-[2.5px] rounded-full transition-all duration-75 ${
                              micVolume > 4
                                ? 'bg-gradient-to-t from-cyan-400 to-emerald-400'
                                : darkMode ? 'bg-cyan-600/35' : 'bg-cyan-300'
                            }`}
                            style={{ height: `${h}px` }}
                          />
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Main input bar */}
                <div className="input-bar flex items-end gap-2 px-3 py-2.5 max-w-[680px] mx-auto">

                  {/* Voice Live Mode */}
                  <button
                    onClick={openVoiceMode}
                    title="EdgeVoice Live Mode"
                    className={`shrink-0 w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-150 cursor-pointer ${
                      darkMode
                        ? 'text-slate-500 hover:text-cyan-400 hover:bg-cyan-500/10 border border-white/[0.06] hover:border-cyan-500/20'
                        : 'text-slate-400 hover:text-cyan-600 bg-white hover:bg-cyan-50 border border-slate-200 hover:border-cyan-200'
                    }`}
                  >
                    <Headset weight="fill" className="w-3.5 h-3.5" />
                  </button>

                  {/* Inline mic toggle */}
                  <button
                    onClick={toggleVoice}
                    title={isListening ? 'Stop recording' : 'Record voice'}
                    className={`shrink-0 w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-150 cursor-pointer ${
                      isListening
                        ? 'bg-rose-500 text-white animate-pulse border border-rose-500'
                        : darkMode
                          ? 'text-slate-500 hover:text-violet-400 hover:bg-violet-500/10 border border-white/[0.06] hover:border-violet-500/20'
                          : 'text-slate-400 hover:text-violet-600 bg-white hover:bg-violet-50 border border-slate-200 hover:border-violet-200'
                    }`}
                  >
                    {isListening
                      ? <MicrophoneSlash weight="fill" className="w-3.5 h-3.5" />
                      : <Microphone weight="fill" className="w-3.5 h-3.5" />
                    }
                  </button>

                  {/* Textarea */}
                  <textarea
                    ref={textareaRef}
                    rows={1}
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onKeyDown={onKeyDown}
                    placeholder={
                      isListening
                        ? 'Listening… speak now'
                        : isTranscribing
                          ? 'Transcribing with Faster-Whisper…'
                          : 'Ask anything — voice, text, image, or PDF'
                    }
                    disabled={isProcessing || isTranscribing}
                    className={`flex-1 resize-none bg-transparent outline-none text-[13px] leading-relaxed max-h-40 py-1 ${
                      darkMode
                        ? 'text-slate-100 placeholder:text-slate-600'
                        : 'text-[#16142e] placeholder:text-slate-400'
                    } disabled:opacity-50`}
                  />

                  {/* File upload */}
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    title="Upload image or PDF"
                    className={`shrink-0 w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-150 cursor-pointer ${
                      darkMode
                        ? 'text-slate-600 hover:text-slate-300 hover:bg-white/[0.05]'
                        : 'text-slate-400 hover:text-violet-600 hover:bg-violet-50'
                    }`}
                  >
                    <ImageIcon weight="bold" className="w-3.5 h-3.5" />
                  </button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*,.pdf,.docx,.txt"
                    className="hidden"
                    onChange={handleFileUpload}
                  />

                  {/* Send button */}
                  <button
                    onClick={() => sendMessage(inputText, imagePreview, imageFile)}
                    disabled={(!inputText.trim() && !imagePreview && !imageFile) || isProcessing}
                    className="shrink-0 w-8 h-8 rounded-lg bg-violet-600 hover:bg-violet-500 active:bg-violet-700 text-white flex items-center justify-center transition-all duration-150 disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer"
                  >
                    <PaperPlaneRight weight="fill" className="w-3.5 h-3.5" />
                  </button>
                </div>

                <p className={`text-center text-[10px] mt-1.5 ${
                  darkMode ? 'text-slate-700' : 'text-slate-400'
                }`}>
                  100% offline · AMD Ryzen 7 · Enter to send · Shift+Enter for new line
                </p>
              </div>

            </div>
          )}

        </div>
      </div>
    </div>
  );
}
