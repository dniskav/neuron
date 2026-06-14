// ─── Flatten Layer ────────────────────────────────────────────────────────────
//
// Converts a 3D tensor [H][W][C] into a flat 1D vector of length H·W·C.
// Acts as the bridge between spatial layers (Conv2D, MaxPool2D) and
// fully-connected layers (NetworkN).
//
// Layout convention (row-major, channels last):
//   flat[h * W * C + w * C + c] = input[h][w][c]
//
// Forward:  number[][][] → number[]   (H·W·C elements)
// Backward: number[]    → number[][][]  (reconstructed from saved shape)
//
// The input shape [H, W, C] is saved during forward and reused in backward.
// It is not meaningful to call backward before forward.
//
// ─────────────────────────────────────────────────────────────────────────────

export class Flatten {
  inputShape: [number, number, number] | null = null;  // [H, W, C]

  // ── Forward pass ──────────────────────────────────────────────────────────
  // Flattens input[h][w][c] into a 1D array in row-major, channel-last order.
  forward(input: number[][][]): number[] {
    const H = input.length;
    const W = input[0].length;
    const C = input[0][0].length;

    this.inputShape = [H, W, C];

    const flat: number[] = new Array(H * W * C);
    let idx = 0;
    for (let h = 0; h < H; h++) {
      for (let w = 0; w < W; w++) {
        for (let c = 0; c < C; c++) {
          flat[idx++] = input[h][w][c];
        }
      }
    }

    return flat;
  }

  // ── Backward pass ─────────────────────────────────────────────────────────
  // Reshapes a flat gradient vector back into [H][W][C] using the saved shape.
  backward(dOutput: number[]): number[][][] {
    if (!this.inputShape) {
      throw new Error('Flatten.backward: call forward() first');
    }

    const [H, W, C] = this.inputShape;

    if (dOutput.length !== H * W * C) {
      throw new Error(
        `Flatten.backward: expected gradient of length ${H * W * C}, got ${dOutput.length}`
      );
    }

    const dInput: number[][][] = Array.from({ length: H }, () =>
      Array.from({ length: W }, () => new Array(C).fill(0))
    );

    let idx = 0;
    for (let h = 0; h < H; h++) {
      for (let w = 0; w < W; w++) {
        for (let c = 0; c < C; c++) {
          dInput[h][w][c] = dOutput[idx++];
        }
      }
    }

    return dInput;
  }
}
