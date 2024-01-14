#include <immintrin.h> // avx
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double get_wtime(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1.0e-6;
}

/// parameters
const size_t N = 1 << 16; // system size
const float eps = 5.0;    // Lenard-Jones, eps
const float rm = 0.1;     // Lenard-Jones, r_m

/// compute the Lennard-Jones force particle at position x0
float compute_force(float *positions, float x0) {
    float rm2 = rm * rm;
    // vectorize rm2
    __m256 rm2_vec = _mm256_set1_ps(rm2);
    //** float force = 0.;
    // vectorize float force;
    __m256 force_vec = _mm256_set1_ps(0.0);
    // vectorize x0
    __m256 x0_vec = _mm256_set1_ps(x0);

    // alias array of aligned float32 to array of 256bit vectors (__mm256)
    __m256 *p_vec = (__m256 *)positions;
    for (size_t i = 0; i < N / 8; ++i) { // N/8 iterations
        //** float r = x0 - positions[i];
        // subtract positions vector from vectorized x0
        __m256 r_vec = _mm256_sub_ps(x0_vec, *(p_vec + i));

        //** mm float r2 = r * r;     // r^2
        // square r_vec by mutiplying with itself
        __m256 r2_vec = _mm256_mul_ps(r_vec, r_vec);
        //** float s2 = rm2 / r2;     // (rm/r)^2
        __m256 s2_vec = _mm256_div_ps(rm2_vec, r2_vec);
        //** float s6 = s2 * s2 * s2; // (rm/r)^6
        __m256 s4_vec = _mm256_mul_ps(s2_vec, s2_vec);
        __m256 s6_vec = _mm256_mul_ps(s4_vec, s2_vec);

        //** force += 12 * eps * (s6 * s6 - s6) / r;
        // since 12, eps are constants divide them and create a vector from them
        __m256 eps12 = _mm256_set1_ps(12 * eps);
        // divide it by r
        __m256 scaling = _mm256_div_ps(eps12, r_vec);
        // fuse multiply subtract
        __m256 fms = _mm256_fmsub_ps(s6_vec, s6_vec, s6_vec);
        // multiply scaling by fms and add to force
        force_vec = _mm256_fmadd_ps(scaling, fms, force_vec);
    }
    // horizontal sum -> horizontal sum -> lane flip -> sum, to get sum of all
    // floats in vector
    force_vec = _mm256_hadd_ps(force_vec, force_vec);
    force_vec = _mm256_hadd_ps(force_vec, force_vec);
    force_vec = _mm256_add_ps(force_vec, _mm256_permute2f128_ps(force_vec, force_vec,1));
    return *((float*) &force_vec);
}

int main(int argc, const char **argv) {
    /// init random number generator
    srand48(1);

    // aligned memory alocation
    float *positions;
    int alloc_status = posix_memalign((void **)&positions, 32, N * sizeof(float));
    if (alloc_status != 0){
        fprintf(stderr, "Aligned alloc failed");
        exit(alloc_status);
    }

    for (size_t i = 0; i < N; i++)
        positions[i] = drand48() + 0.1;

    /// timings
    double start, end;

    float x0[] = {0., -0.1, -0.2};
    float f0[] = {0, 0, 0};

    const size_t repetitions = 1000;
    start = get_wtime();
    for (size_t i = 0; i < repetitions; ++i) {
        for (size_t j = 0; j < 3; ++j)
            f0[j] += compute_force(positions, x0[j]);
    }
    end = get_wtime();

    for (size_t j = 0; j < 3; ++j)
        printf("Force acting at x_0=%lf : %lf\n", x0[j], f0[j] / repetitions);

    printf("elapsed time: %lf mus\n", 1e6 * (end - start));
    return 0;
}
