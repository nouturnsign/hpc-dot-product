// dot_bench.cpp
// Portable C++17 benchmark for dot product variants: naive, unrolled, and OpenMP-parallelized.
// Enhanced with a robust argument parser and thread control.
//
// Example:
//   ./dot_bench --size 10000000 --iters 5 --warmup 2 --threads 8

#include <algorithm>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64) || defined(__AVX__) || defined(__AVX2__)
#include <immintrin.h>
#endif

// ==========================================================
// Simple Argument Parser (dependency-free)
// ==========================================================
struct Args
{
    size_t size = 10'000'000;
    int iters = 7;
    int warmup = 2;
    int threads = 0; // 0 = auto
    bool help = false;
};

class ArgParser
{
public:
    static Args parse(int argc, char** argv)
    {
        Args args;
        std::unordered_map<std::string, std::string> opts;

        for (int i = 1; i < argc; ++i)
        {
            std::string key = argv[i];
            if (key == "-h" || key == "--help")
            {
                args.help = true;
                return args;
            }
            if (key[0] == '-')
            {
                if (i + 1 < argc && argv[i + 1][0] != '-')
                {
                    opts[key] = argv[++i];
                }
                else
                {
                    opts[key] = "true";
                }
            }
        }

        if (opts.count("--size"))
            args.size = std::stoull(opts["--size"]);
        if (opts.count("-n"))
            args.size = std::stoull(opts["-n"]);

        if (opts.count("--iters"))
            args.iters = std::stoi(opts["--iters"]);
        if (opts.count("-i"))
            args.iters = std::stoi(opts["-i"]);

        if (opts.count("--warmup"))
            args.warmup = std::stoi(opts["--warmup"]);
        if (opts.count("-w"))
            args.warmup = std::stoi(opts["-w"]);

        if (opts.count("--threads"))
            args.threads = std::stoi(opts["--threads"]);
        if (opts.count("-t"))
            args.threads = std::stoi(opts["-t"]);

        return args;
    }

    static void print_help(const char* prog)
    {
        std::cout << "Usage: " << prog << " [options]\n\n"
                  << "Options:\n"
                  << "  -n, --size N        Vector length (default: 10,000,000)\n"
                  << "  -i, --iters I       Benchmark iterations (default: 7)\n"
                  << "  -w, --warmup W      Warmup runs (default: 2)\n"
                  << "  -t, --threads T     Number of OpenMP threads (0 = auto)\n"
                  << "  -h, --help          Show this help message\n";
    }
};

// ==========================================================
// Dot Product Implementations
// ==========================================================

// Naive
double dot_naive(const double* a, const double* b, size_t n)
{
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i)
        sum += a[i] * b[i];
    return sum;
}

// Unrolled (4x)
double dot_unrolled(const double* a, const double* b, size_t n)
{
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    size_t i = 0;
    size_t limit = n / 4 * 4;
    for (; i < limit; i += 4)
    {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    for (; i < n; ++i)
        s0 += a[i] * b[i];
    return s0 + s1 + s2 + s3;
}

// OpenMP
#ifdef _OPENMP
double dot_openmp(const double* a, const double* b, size_t n)
{
    double sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
    for (size_t i = 0; i < n; ++i)
        sum += a[i] * b[i];
    return sum;
}
#else
double dot_openmp(const double* a, const double* b, size_t n) { return dot_naive(a, b, n); }
#endif

// SIMD (NEON for ARM, AVX2 for x86)
double dot_simd(const double* a, const double* b, size_t n)
{
#if defined(__aarch64__)
    // ARM NEON implementation
    float64x2_t vsum = vdupq_n_f64(0.0);
    size_t i = 0;
    size_t limit = n / 2 * 2;
    for (; i < limit; i += 2)
    {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        vsum = vaddq_f64(vsum, vmulq_f64(va, vb));
    }
    double buf[2];
    vst1q_f64(buf, vsum);
    double sum = buf[0] + buf[1];
    for (; i < n; ++i)
        sum += a[i] * b[i];
    return sum;

#elif defined(__AVX__) || defined(__AVX2__)
    // x86 AVX2 version
    __m256d vsum = _mm256_setzero_pd();
    size_t i = 0;
    size_t limit = n / 4 * 4;
    for (; i < limit; i += 4)
    {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        __m256d prod = _mm256_mul_pd(va, vb);
        vsum = _mm256_add_pd(vsum, prod);
    }
    alignas(32) double buf[4];
    _mm256_store_pd(buf, vsum);
    double sum = buf[0] + buf[1] + buf[2] + buf[3];
    for (; i < n; ++i)
        sum += a[i] * b[i];
    return sum;

#else
    // fallback if SIMD not supported
    return dot_unrolled(a, b, n);
#endif
}

#ifdef _OPENMP
double dot_openmp_simd(const double* a, const double* b, size_t n)
{
    double sum = 0.0;
#pragma omp parallel for simd reduction(+ : sum)
    for (size_t i = 0; i < n; ++i)
        sum += a[i] * b[i];

    return sum;
}
#else
double dot_openmp_simd(const double* a, const double* b, size_t n) { return dot_simd(a, b, n); }
#endif

// ==========================================================
// Benchmarking Utilities
// ==========================================================
struct BenchmarkResult
{
    std::string name;
    double value;
    double time_s;
    double gflops;
};

using DotFn = std::function<double(const double*, const double*, size_t)>;

static void fill_random(double* a, size_t n, unsigned seed = 1234)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < n; ++i)
        a[i] = dist(rng);
}

static double time_single_run(const DotFn& fn, const double* a, const double* b, size_t n)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    volatile double val = fn(a, b, n);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    return dt.count();
}

static BenchmarkResult benchmark_fn(
    const std::string& name, const DotFn& fn, const double* a, const double* b, size_t n, int warmup, int iters)
{
    for (int i = 0; i < warmup; ++i)
        fn(a, b, n);

    std::vector<double> times;
    times.reserve(iters);
    double val = fn(a, b, n);

    for (int i = 0; i < iters; ++i)
        times.push_back(time_single_run(fn, a, b, n));

    std::sort(times.begin(), times.end());
    double median = times[times.size() / 2];
    double gflops = (2.0 * n) / (median * 1e9);
    return { name, val, median, gflops };
}

// ==========================================================
// Main
// ==========================================================
int main(int argc, char** argv)
{
    Args args = ArgParser::parse(argc, argv);
    if (args.help)
    {
        ArgParser::print_help(argv[0]);
        return 0;
    }

#ifdef _OPENMP
    if (args.threads > 0)
    {
        omp_set_num_threads(args.threads);
    }
#endif

    std::cout << "Dot Product Benchmark\n";
    std::cout << "---------------------\n";
    std::cout << "Size:    " << args.size << "\n";
    std::cout << "Iters:   " << args.iters << "\n";
    std::cout << "Warmup:  " << args.warmup << "\n";
#ifdef _OPENMP
    std::cout << "Threads: " << (args.threads > 0 ? args.threads : omp_get_max_threads()) << "\n";
#else
    std::cout << "Threads: 1 (OpenMP disabled)\n";
#endif
    std::cout << "\n";

    std::vector<double> a(args.size), b(args.size);
    fill_random(a.data(), args.size, 123);
    fill_random(b.data(), args.size, 456);

    std::vector<std::pair<std::string, DotFn>> tests = { { "naive", dot_naive }, { "unrolled", dot_unrolled },
        { "openmp", dot_openmp }, { "simd", dot_simd }, { "openmp_simd", dot_openmp_simd } };

    std::cout << std::left << std::setw(12) << "Variant" << std::setw(12) << "Time (s)" << std::setw(12) << "GFLOPS"
              << "Result\n";
    std::cout << std::string(50, '-') << "\n";

    for (auto& p : tests)
    {
        auto res = benchmark_fn(p.first, p.second, a.data(), b.data(), args.size, args.warmup, args.iters);
        std::cout << std::left << std::setw(12) << res.name << std::setw(12) << std::fixed << std::setprecision(6)
                  << res.time_s << std::setw(12) << std::fixed << std::setprecision(3) << res.gflops << std::scientific
                  << std::setprecision(6) << res.value << "\n";
    }

    return 0;
}
