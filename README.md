# RV-Sparse: Sparse Matrix–Vector Multiplication

Implementation of the `sparse_multiply` coding challenge.  
Converts a dense row-major matrix to **Compressed Sparse Row (CSR)** format and computes **y = A · x** — with **zero dynamic memory allocation** inside the function.

---

***Github Link*** : https://github.com/abhayr2/RISC-V-Sparse-Task

## What it does

`sparse_multiply` performs three things in a single call:

1. **Scans** a row-major matrix `A` for non-zero elements.
2. **Extracts** them into CSR format using caller-provided buffers (`values`, `col_indices`, `row_ptrs`).
3. **Computes** the matrix–vector product `y = A * x` and writes results into the caller-provided output buffer `y`.

### Function Signature

```c
void sparse_multiply(
    int           rows,
    int           cols,
    const double *A,           // input matrix, row-major [rows x cols]
    const double *x,           // input vector [cols]
    int          *out_nnz,     // written with the number of non-zeros found
    double       *values,      // CSR values buffer       — capacity >= rows*cols
    int          *col_indices, // CSR column index buffer  — capacity >= rows*cols
    int          *row_ptrs,    // CSR row pointer buffer   — capacity >= rows+1
    double       *y            // output vector [rows]    — written by this function
);
```

### CSR Format

| Buffer        | Length     | Meaning                                                            |
|---------------|------------|--------------------------------------------------------------------|
| `values`      | `nnz`      | Non-zero values in row-major scan order                            |
| `col_indices` | `nnz`      | Column index of each corresponding value                           |
| `row_ptrs`    | `rows + 1` | `row_ptrs[i]` = start of row `i` in `values`; `row_ptrs[rows] = nnz` |

---

## How to build and run

```bash
gcc -o run challenge.c -lm
./run
```

> **Note:** `-lm` must come *after* `challenge.c` on GCC/Linux — the linker resolves symbols left-to-right.

Expected output:

```
Iter  0 [ 17x 37, density=0.38, nnz= 231]: PASS (Max error: 0.00e+00)
...
All tests passed! (100/100 iterations passed)
```

---

## Implementation

```c
void sparse_multiply(
    int rows, int cols, const double* A, const double* x,
    int* out_nnz, double* values, int* col_indices, int* row_ptrs,
    double* y
) {
    int nnz = 0;

    /* Phase 1: build CSR — one row-major pass, O(rows * cols) */
    for (int i = 0; i < rows; i++) {
        row_ptrs[i] = nnz;
        for (int j = 0; j < cols; j++) {
            double v = A[i * cols + j];
            if (v != 0.0) {
                values[nnz]      = v;
                col_indices[nnz] = j;
                nnz++;
            }
        }
    }
    row_ptrs[rows] = nnz;
    *out_nnz = nnz;

    /* Phase 2: CSR matvec — O(nnz) */
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        for (int k = row_ptrs[i]; k < row_ptrs[i + 1]; k++) {
            sum += values[k] * x[col_indices[k]];
        }
        y[i] = sum;
    }
}
```

**Zero allocation guarantee** — `sparse_multiply` calls no allocator. All buffers are owned by the caller (the test harness uses `malloc`; a bare-metal caller could use stack or static arrays).

**Two-phase design** — Phase 1 extracts non-zeros in a single row-major scan. Phase 2 computes the matvec purely over non-zero entries, skipping all zeros.

---

## Test harness

The harness runs **100 random iterations**, each with:
- Random matrix dimensions: rows and cols each in `[5, 45]`
- Random density: `[0.05, 0.40]`
- Random values in `[-10, 10]` for both `A` and `x`
- Correctness check against a naive dense matvec reference with mixed absolute/relative tolerance (`1e-7 + 1e-7 * |y_ref[i]|`)

All 100/100 iterations pass with max error `0.00e+00`.

---

## Project structure

```
rv-sparse/
├── challenge.c   # sparse_multiply implementation + official test harness
└── README.md
```
***NOTE*** : Claude was used to help with the README.md and make the code more readable, like comments, slight fixes.
