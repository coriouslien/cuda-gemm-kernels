// g++ -O2 -std=c++17 online_softmax.cpp -o online_softmax && ./online_softmax
#include <cmath>
#include <cstdio>
#include <limits>

// Reference: safe softmax over the whole row (Lesson 2) — three passes
void safe_softmax(const float* x, float* p, int n) {
    float m = x[0];
    for (int i = 1; i < n; ++i) m = std::fmax(m, x[i]);          // pass 1: max
    float l = 0.f;
    for (int i = 0; i < n; ++i) l += std::exp(x[i] - m);         // pass 2: sum
    for (int i = 0; i < n; ++i) p[i] = std::exp(x[i] - m) / l;   // pass 3: weights
}

// Online softmax: visit the row block by block, keep only m and l
void online_softmax(const float* x, float* p, int n, int B) {
    float m = -std::numeric_limits<float>::infinity();
    float l = 0.f;
    for (int start = 0; start < n; start += B) {
        int end = start + B < n ? start + B : n;
        float m_blk = x[start];
        for (int i = start + 1; i < end; ++i) m_blk = std::fmax(m_blk, x[i]);
        float m_new = std::fmax(m, m_blk);
        float scale = std::exp(m - m_new);            // correction for the old sum
        float s = 0.f;
        for (int i = start; i < end; ++i) s += std::exp(x[i] - m_new);
        l = l * scale + s;
        std::printf("block [%d,%d): m_blk=%g m_new=%g scale=%.4f block_sum=%.4f l=%.4f\n",
                    start, end, m_blk, m_new, scale, s, l);
        m = m_new;
    }
    for (int i = 0; i < n; ++i) p[i] = std::exp(x[i] - m) / l;   // final pass (removed in Lesson 6)
}

int main() {
    const int n = 6;
    float x[n] = {2, 1, 0, 3, 1, 4};
    float p_ref[n], p_on[n];
    safe_softmax(x, p_ref, n);
    online_softmax(x, p_on, n, 2);
    float maxdiff = 0.f;
    for (int i = 0; i < n; ++i) {
        std::printf("x=%g  ref=%.4f  online=%.4f\n", x[i], p_ref[i], p_on[i]);
        maxdiff = std::fmax(maxdiff, std::fabs(p_ref[i] - p_on[i]));
    }
    std::printf("max |ref - online| = %g\n", maxdiff);
}
    

