// -------------------------------------------------------------------------
// DataTable.jsx — Collapsible tabular view of chart data.
//
// Accepts columns (array of {key, label}) and rows (array of objects).
// Toggled open/closed via a button so it doesn't clutter the chart.
// -------------------------------------------------------------------------
import { useState } from "react";

export default function DataTable({ columns, rows, title }) {
  const [open, setOpen] = useState(false);

  if (!rows || rows.length === 0) return null;

  return (
    <div className="rounded-lg bg-[#101014] border border-s2-border">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center justify-between w-full px-3 py-2 text-[11px] uppercase tracking-widest text-s2-muted hover:text-s2-text transition-colors"
      >
        <span>{title || "View Data Table"}</span>
        <span className="font-mono text-s2-blue">{open ? "[-]" : "[+]"}</span>
      </button>

      {open && (
        <div className="overflow-x-auto max-h-[320px] overflow-y-auto border-t border-s2-border">
          <table className="w-full text-[11px] font-mono">
            <thead className="sticky top-0 bg-[#101014]">
              <tr className="text-s2-muted text-left border-b border-s2-border">
                {columns.map((col) => (
                  <th key={col.key} className="px-2.5 py-1.5 whitespace-nowrap">
                    {col.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="text-s2-text">
              {rows.map((row, i) => (
                <tr
                  key={i}
                  className={i % 2 === 0 ? "bg-[#101014]" : "bg-[#0c0c0f]"}
                >
                  {columns.map((col) => (
                    <td key={col.key} className="px-2.5 py-1 whitespace-nowrap">
                      {typeof row[col.key] === "number"
                        ? row[col.key].toFixed(col.decimals ?? 2)
                        : row[col.key] ?? "--"}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
