// ─── LOSS PLOTTER (ASCII) ─────────────────────────────────────────────────────
//
// Renders a training loss curve as an ASCII chart in the terminal.
// No external dependencies — pure string manipulation.
//
// Characters used:
//   │  ─  ┼  ┤  ┬   — box-drawing for axes and ticks
//   *                — data point marker
//   ·                — background grid dot (optional)
//
// Layout:
//
//   ┌─ Title ──────────────────────────────────────────────┐
//   │                                                       │
//   max ┤  *                                               │
//       │    *  *                                           │
//       │        *  *                                       │
//       │              * * *                                │
//   min ┤                    * * * * ·                      │
//       └───────────────────────────────────────────────── epoch
//
// Usage:
//
//   const plotter = new LossPlotter({ width: 60, height: 15, title: 'Training Loss' });
//   for (let e = 0; e < epochs; e++) {
//     const loss = trainEpoch(...);
//     plotter.add(loss, e);
//   }
//   plotter.print();
//

// ─── LossPlotter ─────────────────────────────────────────────────────────────

export class LossPlotter {
  private readonly width: number;
  private readonly height: number;
  private readonly title: string;
  private losses: number[];
  private epochs: number[];

  constructor(options?: {
    width?: number;
    height?: number;
    title?: string;
  }) {
    const width  = options?.width  ?? 60;
    const height = options?.height ?? 15;
    if (!Number.isInteger(width) || width <= 0) {
      throw new Error(`LossPlotter.constructor: width must be a positive integer, got ${width}`);
    }
    if (!Number.isInteger(height) || height <= 0) {
      throw new Error(`LossPlotter.constructor: height must be a positive integer, got ${height}`);
    }
    this.width  = width;
    this.height = height;
    this.title  = options?.title  ?? 'Loss Curve';
    this.losses = [];
    this.epochs = [];
  }

  // Add a single (loss, epoch) pair.
  add(loss: number, epoch?: number): void {
    if (!Number.isFinite(loss)) {
      throw new Error(`LossPlotter.add: loss must be a finite number, got ${loss}`);
    }
    this.losses.push(loss);
    this.epochs.push(epoch ?? this.losses.length - 1);
  }

  // Add multiple loss values (epochs are auto-numbered from 0).
  addMultiple(losses: number[]): void {
    for (let i = 0; i < losses.length; i++) {
      if (!Number.isFinite(losses[i])) {
        throw new Error(`LossPlotter.addMultiple: losses[${i}] must be a finite number, got ${losses[i]}`);
      }
    }
    for (const l of losses) this.add(l);
  }

  // Returns the ASCII plot as a multi-line string.
  render(): string {
    if (this.losses.length === 0) return `[${this.title}] — no data yet`;

    const losses = this.losses;
    let minL = losses[0];
    let maxL = losses[0];
    for (let i = 1; i < losses.length; i++) {
      if (losses[i] < minL) minL = losses[i];
      if (losses[i] > maxL) maxL = losses[i];
    }
    const range  = maxL - minL || 1;

    // Width of the Y-axis label column (e.g. "0.1234 ┤"): at least 8 characters,
    // and wide enough for the longest label that can be drawn (negative or 7-char
    // values such as "-0.2500" or "0.01417" would otherwise push the axis out of line).
    const midL = (minL + maxL) / 2;
    const longestLabel = Math.max(_fmtNum(maxL).length, _fmtNum(minL).length, _fmtNum(midL).length);
    const yLabelW = Math.max(8, longestLabel + 2);
    // Actual plot area width (inside the axes); the whole block is `width` wide
    const plotW   = Math.max(4, this.width - yLabelW - 1);
    const plotH   = Math.max(3, this.height);

    // Create a 2D character grid: rows × cols (all spaces initially)
    const grid: string[][] = Array.from({ length: plotH }, () =>
      new Array(plotW).fill(' '),
    );

    // Map each loss value to a grid column via interpolation
    const n = losses.length;
    for (let idx = 0; idx < n; idx++) {
      // col: spread losses evenly across plotW columns
      const col = Math.round((idx / Math.max(1, n - 1)) * (plotW - 1));
      // row: 0 = top (max), plotH-1 = bottom (min)
      const norm = (losses[idx] - minL) / range;       // 0 = min, 1 = max
      const row  = Math.round((1 - norm) * (plotH - 1));
      grid[row][col] = '*';
    }

    // Build the output lines
    const lines: string[] = [];

    // Title line: '┌' + inner + '┐', where inner has the same width as the rows
    // between their first and last column (yLabelW + plotW - 1 characters).
    // A title longer than that is cut so the corner stays in the border column.
    const inner = yLabelW + plotW - 1;
    let titlePadded = [...` ${this.title} `];
    if (titlePadded.length > inner) titlePadded = titlePadded.slice(0, inner);
    const dashCount = inner - titlePadded.length;
    lines.push('┌' + titlePadded.join('') + '─'.repeat(dashCount) + '┐');

    // Plot rows with Y-axis labels
    for (let row = 0; row < plotH; row++) {
      let label: string;
      if (row === 0) {
        label = _fmtNum(maxL).padStart(yLabelW - 2) + ' ┤';
      } else if (row === plotH - 1) {
        label = _fmtNum(minL).padStart(yLabelW - 2) + ' ┤';
      } else if (row === Math.floor(plotH / 2)) {
        label = _fmtNum(midL).padStart(yLabelW - 2) + ' ┤';
      } else {
        label = ' '.repeat(yLabelW - 1) + '│';
      }
      lines.push(label + grid[row].join('') + '│');
    }

    // X-axis bottom line
    const xAxis = ' '.repeat(yLabelW - 1) + '└' + '─'.repeat(plotW) + '┘';
    lines.push(xAxis);

    // X-axis epoch labels: first and last epoch
    const firstEpoch = String(this.epochs[0]);
    const lastEpoch  = String(this.epochs[this.epochs.length - 1]);
    const xLabel = ' '.repeat(yLabelW) +
      firstEpoch +
      ' '.repeat(Math.max(1, plotW - firstEpoch.length - lastEpoch.length)) +
      lastEpoch + '  epoch';
    lines.push(xLabel);

    // Footer: min / max / last loss values
    lines.push(
      ` min=${_fmtNum(minL)}  max=${_fmtNum(maxL)}  last=${_fmtNum(losses[losses.length - 1])}  n=${n}`,
    );

    return lines.join('\n');
  }

  // Prints the chart to stdout.
  print(): void {
    console.log(this.render());
  }

  // Clears all accumulated data.
  reset(): void {
    this.losses = [];
    this.epochs = [];
  }
}

// ─── Helper ───────────────────────────────────────────────────────────────────

// Formats a number into at most 6 characters for axis labels.
function _fmtNum(n: number): string {
  if (Math.abs(n) >= 1e4 || (Math.abs(n) < 1e-3 && n !== 0)) {
    return n.toExponential(1);
  }
  return n.toPrecision(4);
}
