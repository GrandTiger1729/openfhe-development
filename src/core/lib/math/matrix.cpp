//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2014-2023, NJIT, Duality Technologies Inc. and other contributors
//
// All rights reserved.
//
// Author TPOC: contact@openfhe.org
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//==================================================================================

/*
  matrix class implementations and type specific implementations
*/

#ifndef LBCRYPTO_LIB_MATH_MATRIX_CPP
#define LBCRYPTO_LIB_MATH_MATRIX_CPP

#include "math/math-hal.h"
#include "math/matrix-impl.h"

#include "utils/exception.h"
#include "utils/parallel.h"

#include <vector>
#include <cstddef>
#include <cmath>
#if defined(__AVX2__)
  #include <immintrin.h>
#endif

// this is the implementation of matrices of things that are in core
// and that need template specializations

namespace lbcrypto {

#define MODEQ_FOR_TYPE(T)                             \
    template <>                                       \
    Matrix<T>& Matrix<T>::ModEq(const T& element) {   \
        for (size_t row = 0; row < rows; ++row) {     \
            for (size_t col = 0; col < cols; ++col) { \
                data[row][col].ModEq(element);        \
            }                                         \
        }                                             \
        return *this;                                 \
    }

MODEQ_FOR_TYPE(NativeInteger)
MODEQ_FOR_TYPE(BigInteger)

#define MODSUBEQ_FOR_TYPE(T)                                               \
    template <>                                                            \
    Matrix<T>& Matrix<T>::ModSubEq(Matrix<T> const& b, const T& element) { \
        for (size_t row = 0; row < rows; ++row) {                          \
            for (size_t col = 0; col < cols; ++col) {                      \
                data[row][col].ModSubEq(b.data[row][col], element);        \
            }                                                              \
        }                                                                  \
        return *this;                                                      \
    }

MODSUBEQ_FOR_TYPE(NativeInteger)
MODSUBEQ_FOR_TYPE(BigInteger)

// -----------------------------------------------------------------------------
// SIMD Cholesky (int32 -> double) with scalar fallback
// Assumption: input is symmetric positive definite; result is lower-triangular.
// -----------------------------------------------------------------------------

static inline void CholeskyKernelSIMD_RowMajor(double* A, std::size_t n) {
    std::vector<double> colk(n);

    for (std::size_t k = 0; k < n; ++k) {
        // diagonal
        double akk = std::sqrt(A[k * n + k]);
        A[k * n + k] = akk;

        // normalize column k below diagonal; zero upper triangle of row k
        for (std::size_t i = k + 1; i < n; ++i) {
            A[i * n + k] /= akk;
            A[k * n + i]  = 0.0;
        }

        // cache column k (contiguous)
        colk[k] = akk; // diag kept; not used in updates
        for (std::size_t i = k + 1; i < n; ++i) {
            colk[i] = A[i * n + k];
        }

        // trailing update: for j = k+1..i, A[i,j] -= A[i,k] * A[j,k]
        for (std::size_t i = k + 1; i < n; ++i) {
            const double aik = colk[i];

        #if defined(__AVX2__)
            __m256d aik_v = _mm256_set1_pd(aik);
            std::size_t j = k + 1;

            // vectorized along row i: process 4 doubles per iter
            for (; j + 3 < i + 1; j += 4) {
                __m256d ck_v  = _mm256_loadu_pd(&colk[j]);
                __m256d aij_v = _mm256_loadu_pd(&A[i * n + j]);
                aij_v = _mm256_sub_pd(aij_v, _mm256_mul_pd(aik_v, ck_v));
                _mm256_storeu_pd(&A[i * n + j], aij_v);
            }
            // scalar tail
            for (; j <= i; ++j) {
                A[i * n + j] -= aik * colk[j];
            }
        #else
            // pure scalar fallback
            for (std::size_t j = k + 1; j <= i; ++j) {
                A[i * n + j] -= aik * colk[j];
            }
        #endif
        }
    }
}

inline Matrix<double> CholeskySIMD(const Matrix<int32_t>& input) {
    const std::size_t n = input.GetRows();
    if (n != input.GetCols()) {
        OPENFHE_THROW("not square");
    }

    // copy Matrix<int32_t> -> contiguous row-major double buffer
    std::vector<double> buf(n * n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            buf[i * n + j] = static_cast<double>(input(i, j));
        }
    }

    // run kernel
    CholeskyKernelSIMD_RowMajor(buf.data(), n);

    // copy back to Matrix<double>
    Matrix<double> result([]() { return 0.0; }, n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            result(i, j) = buf[i * n + j];
        }
    }
    return result;
}

// Public APIs (same signatures as original)
Matrix<double> Cholesky(const Matrix<int32_t>& input) {
    return CholeskySIMD(input);
}

void Cholesky(const Matrix<int32_t>& input, Matrix<double>& result) {
    if (input.GetRows() != input.GetCols()) {
        OPENFHE_THROW("not square");
    }
    result = CholeskySIMD(input);
}

// -----------------------------------------------------------------------------
// Convert from Z_q to [-q/2, q/2]
// -----------------------------------------------------------------------------
Matrix<int32_t> ConvertToInt32(const Matrix<BigInteger>& input, const BigInteger& modulus) {
    size_t rows = input.GetRows();
    size_t cols = input.GetCols();
    BigInteger negativeThreshold(modulus / BigInteger(2));
    Matrix<int32_t> result([]() { return 0; }, rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (input(i, j) > negativeThreshold) {
                result(i, j) = -1 * (modulus - input(i, j)).ConvertToInt();
            }
            else {
                result(i, j) = input(i, j).ConvertToInt();
            }
        }
    }
    return result;
}

Matrix<int32_t> ConvertToInt32(const Matrix<BigVector>& input, const BigInteger& modulus) {
    size_t rows = input.GetRows();
    size_t cols = input.GetCols();
    BigInteger negativeThreshold(modulus / BigInteger(2));
    Matrix<int32_t> result([]() { return 0; }, rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            const BigInteger& elem = input(i, j).at(0);
            if (elem > negativeThreshold) {
                result(i, j) = -1 * (modulus - elem).ConvertToInt();
            }
            else {
                result(i, j) = elem.ConvertToInt();
            }
        }
    }
    return result;
}

// Keep explicit instantiations only in one TU to avoid ODR issues.
// If another TU already instantiates these, remove the lines below.
template class Matrix<double>;
template class Matrix<int>;
template class Matrix<int64_t>;

}  // namespace lbcrypto

#endif // LBCRYPTO_LIB_MATH_MATRIX_CPP