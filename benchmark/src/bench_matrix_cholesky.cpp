#include <benchmark/benchmark.h>
#include <random>
#include "math/matrix-impl.h"

using lbcrypto::Matrix;

// Forward decl of your public Cholesky API (matrix.cpp)
namespace lbcrypto {
    Matrix<double> Cholesky(const Matrix<int32_t>&);
}

// Generate SPD int32 matrix: A = B^T B + n*I (kept small to avoid overflow)
static Matrix<int32_t> MakeSPD(std::size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> dist(-3, 3);

    Matrix<int32_t> B([](){ return 0; }, n, n);
    for (size_t i=0;i<n;++i)
        for (size_t j=0;j<n;++j)
            B(i,j) = dist(rng);

    Matrix<int32_t> A([](){ return 0; }, n, n);
    // A = B^T B
    for (size_t i=0;i<n;++i) {
        for (size_t j=0;j<=i;++j) {
            long long sum = 0;
            for (size_t k=0;k<n;++k)
                sum += (long long)B(k,i) * (long long)B(k,j);
            // symmetrize and keep ints small-ish
            A(i,j) = A(j,i) = static_cast<int32_t>(sum);
        }
    }
    // Make strictly SPD by adding n*I
    for (size_t i=0;i<n;++i)
        A(i,i) += static_cast<int32_t>(n);
    return A;
}

static void BM_Cholesky_N(benchmark::State& st) {
    const int n = static_cast<int>(st.range(0));
    auto A = MakeSPD(n, /*seed*/12345);

    for (auto _ : st) {
        auto L = lbcrypto::Cholesky(A);
        benchmark::DoNotOptimize(L); // prevent DCE
    }

    // Report N^3-ish work as items processed
    st.SetItemsProcessed(static_cast<int64_t>(st.iterations()) * n * n * n);
}
BENCHMARK(BM_Cholesky_N)->Arg(64)->Arg(96)->Arg(128)->Arg(192)->Arg(256);

BENCHMARK_MAIN();
