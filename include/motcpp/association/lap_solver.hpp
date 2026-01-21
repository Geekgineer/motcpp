// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#ifndef TRACKERS_ASSOCIATION_LAP_SOLVER_HPP
#define TRACKERS_ASSOCIATION_LAP_SOLVER_HPP

#include <Eigen/Dense>
#include <vector>
#include <limits>
#include <cstring>
#include <cstdlib>

namespace trackers {
namespace association {

// Internal LAP solver implementation (Jonker-Volgenant algorithm)
namespace lapjv_internal {
typedef signed int int_t;
typedef unsigned int uint_t;
typedef double cost_t;
typedef char boolean;

#ifndef LARGE
#define LARGE 1000000
#endif
#ifndef NEW
#define NEW(x, t, n) if (((x) = (t *)std::malloc(sizeof(t) * (n))) == 0) { return -1; }
#endif
#ifndef FREE
#define FREE(x) if ((x) != 0) { std::free((x)); (x) = 0; }
#endif
#ifndef SWAP_INDICES
#define SWAP_INDICES(a, b) { int_t _temp_index = (a); (a) = (b); (b) = _temp_index; }
#endif

static inline int_t _ccrrt_dense(const uint_t n, cost_t *cost[], int_t *free_rows, int_t *x, int_t *y, cost_t *v) {
    int_t n_free_rows; boolean *unique; 
    for (uint_t i = 0; i < n; i++) { x[i] = -1; v[i] = LARGE; y[i] = 0; }
    for (uint_t i = 0; i < n; i++) { 
        for (uint_t j = 0; j < n; j++) { 
            const cost_t c = cost[i][j]; 
            if (c < v[j]) { v[j] = c; y[j] = (int_t)i; } 
        } 
    }
    NEW(unique, boolean, n); 
    std::memset(unique, 1, n);
    { 
        int_t j = (int_t)n; 
        do { 
            j--; 
            const int_t i = y[j]; 
            if (x[i] < 0) { x[i] = j; } 
            else { unique[i] = 0; y[j] = -1; } 
        } while (j > 0); 
    }
    n_free_rows = 0; 
    for (uint_t i = 0; i < n; i++) { 
        if (x[i] < 0) { free_rows[n_free_rows++] = i; } 
        else if (unique[i]) { 
            const int_t j = x[i]; 
            cost_t min = LARGE; 
            for (uint_t j2 = 0; j2 < n; j2++) { 
                if (j2 == (uint_t)j) continue; 
                const cost_t c = cost[i][j2] - v[j2]; 
                if (c < min) { min = c; } 
            } 
            v[j] -= min; 
        } 
    }
    FREE(unique); 
    return n_free_rows;
}

static inline int_t _carr_dense(const uint_t n, cost_t *cost[], const uint_t n_free_rows, int_t *free_rows, int_t *x, int_t *y, cost_t *v) {
    uint_t current = 0; 
    int_t new_free_rows = 0; 
    uint_t rr_cnt = 0; 
    while (current < n_free_rows) { 
        int_t i0; 
        int_t j1, j2; 
        cost_t v1, v2, v1_new; 
        boolean v1_lowers; 
        rr_cnt++; 
        const int_t free_i = free_rows[current++]; 
        j1 = 0; 
        v1 = cost[free_i][0] - v[0]; 
        j2 = -1; 
        v2 = LARGE; 
        for (uint_t j = 1; j < n; j++) { 
            const cost_t c = cost[free_i][j] - v[j]; 
            if (c < v2) { 
                if (c >= v1) { v2 = c; j2 = (int_t)j; } 
                else { v2 = v1; v1 = c; j2 = j1; j1 = (int_t)j; } 
            } 
        } 
        i0 = y[j1]; 
        v1_new = v[j1] - (v2 - v1); 
        v1_lowers = v1_new < v[j1]; 
        if (rr_cnt < current * n) { 
            if (v1_lowers) { v[j1] = v1_new; } 
            else if (i0 >= 0 && j2 >= 0) { j1 = j2; i0 = y[j2]; } 
            if (i0 >= 0) { 
                if (v1_lowers) { free_rows[--current] = i0; } 
                else { free_rows[new_free_rows++] = i0; } 
            } 
        } else { 
            if (i0 >= 0) { free_rows[new_free_rows++] = i0; } 
        } 
        x[free_i] = j1; 
        y[j1] = free_i; 
    }
    return new_free_rows;
}

static inline uint_t _find_dense(const uint_t n, uint_t lo, cost_t *d, int_t *cols, int_t *y) {
    uint_t hi = lo + 1; 
    cost_t mind = d[cols[lo]]; 
    for (uint_t k = hi; k < n; k++) { 
        int_t j = cols[k]; 
        if (d[j] <= mind) { 
            if (d[j] < mind) { hi = lo; mind = d[j]; } 
            cols[k] = cols[hi]; 
            cols[hi++] = j; 
        } 
    } 
    return hi;
}

static inline int_t _scan_dense(const uint_t n, cost_t *cost[], uint_t *plo, uint_t *phi, cost_t *d, int_t *cols, int_t *pred, int_t *y, cost_t *v) {
    uint_t lo = *plo; 
    uint_t hi = *phi; 
    cost_t h, cred_ij; 
    while (lo != hi) { 
        int_t j = cols[lo++]; 
        const int_t i = y[j]; 
        const cost_t mind = d[j]; 
        h = cost[i][j] - v[j] - mind; 
        for (uint_t k = hi; k < n; k++) { 
            j = cols[k]; 
            cred_ij = cost[i][j] - v[j] - h; 
            if (cred_ij < d[j]) { 
                d[j] = cred_ij; 
                pred[j] = i; 
                if (cred_ij == mind) { 
                    if (y[j] < 0) { return j; } 
                    cols[k] = cols[hi]; 
                    cols[hi++] = j; 
                } 
            } 
        } 
    } 
    *plo = lo; 
    *phi = hi; 
    return -1;
}

static inline int_t find_path_dense(const uint_t n, cost_t *cost[], const int_t start_i, int_t *y, cost_t *v, int_t *pred) {
    uint_t lo = 0, hi = 0; 
    int_t final_j = -1; 
    uint_t n_ready = 0; 
    int_t *cols; 
    cost_t *d; 
    NEW(cols, int_t, n); 
    NEW(d, cost_t, n); 
    for (uint_t i = 0; i < n; i++) { 
        cols[i] = (int_t)i; 
        pred[i] = start_i; 
        d[i] = cost[start_i][i] - v[i]; 
    } 
    while (final_j == -1) { 
        if (lo == hi) { 
            n_ready = lo; 
            hi = _find_dense(n, lo, d, cols, y); 
            for (uint_t k = lo; k < hi; k++) { 
                const int_t j = cols[k]; 
                if (y[j] < 0) { final_j = j; } 
            } 
        } 
        if (final_j == -1) { 
            final_j = _scan_dense(n, cost, &lo, &hi, d, cols, pred, y, v); 
        } 
    } 
    { 
        const cost_t mind = d[cols[lo]]; 
        for (uint_t k = 0; k < n_ready; k++) { 
            const int_t j = cols[k]; 
            v[j] += d[j] - mind; 
        } 
    } 
    FREE(cols); 
    FREE(d); 
    return final_j;
}

static inline int_t _ca_dense(const uint_t n, cost_t *cost[], const int_t n_free_rows, int_t *free_rows, int_t *x, int_t *y, cost_t *v) {
    int_t *pred; 
    NEW(pred, int_t, n); 
    for (int_t *pfree_i = free_rows; pfree_i < free_rows + (int_t)n_free_rows; pfree_i++) { 
        int_t i = -1, j; 
        uint_t k = 0; 
        j = find_path_dense(n, cost, *pfree_i, y, v, pred); 
        while (i != *pfree_i) { 
            i = pred[j]; 
            y[j] = i; 
            SWAP_INDICES(j, x[i]); 
            k++; 
        } 
    } 
    FREE(pred); 
    return 0;
}

static inline int lapjv_internal(const uint_t n, cost_t *cost[], int_t *x, int_t *y) {
    int ret; 
    int_t *free_rows; 
    cost_t *v; 
    NEW(free_rows, int_t, n); 
    NEW(v, cost_t, n); 
    ret = _ccrrt_dense(n, cost, free_rows, x, y, v); 
    int i = 0; 
    while (ret > 0 && i < 2) { 
        ret = _carr_dense(n, cost, (uint_t)ret, free_rows, x, y, v); 
        i++; 
    } 
    if (ret > 0) { 
        ret = _ca_dense(n, cost, (uint_t)ret, free_rows, x, y, v); 
    } 
    FREE(v); 
    FREE(free_rows); 
    return ret;
}

} // namespace lapjv_internal

/**
 * @brief Solves Linear Assignment Problem using Jonker-Volgenant algorithm
 * 
 * Finds optimal assignment between two sets given a cost matrix.
 * This is the core algorithm used for data association in tracking.
 */
class LAPSolver {
public:
    /**
     * @brief Solves linear assignment problem
     * @param costMatrix Cost matrix (n x m) where costMatrix(i,j) is cost of matching i to j
     * @param costLimit Maximum cost threshold for valid matches
     * @param matches Output: vector of [i, j] pairs representing matches
     * @param unmatched_a Output: indices from first set that remain unmatched
     * @param unmatched_b Output: indices from second set that remain unmatched
     */
    static void linearAssignment(
        const Eigen::MatrixXd& costMatrix,
        double costLimit,
        std::vector<std::vector<int>>& matches,
        std::vector<int>& unmatched_a,
        std::vector<int>& unmatched_b) {
        
        matches.clear();
        unmatched_a.clear();
        unmatched_b.clear();
        
        int n = static_cast<int>(costMatrix.rows());
        int m = static_cast<int>(costMatrix.cols());
        
        if (n == 0 || m == 0) {
            for (int i = 0; i < n; ++i) unmatched_a.push_back(i);
            for (int i = 0; i < m; ++i) unmatched_b.push_back(i);
            return;
        }
        
        std::vector<int> x, y;
        lapjv(costMatrix, costLimit, x, y);
        
        // Extract matches and unmatched
        for (int i = 0; i < n; ++i) {
            if (x[i] < 0) {
                unmatched_a.push_back(i);
            } else {
                matches.push_back({i, x[i]});
            }
        }
        
        for (int j = 0; j < m; ++j) {
            if (y[j] < 0) unmatched_b.push_back(j);
        }
    }
    
private:
    static void lapjv(const Eigen::MatrixXd& costMatrix, double costLimit, 
                      std::vector<int>& x, std::vector<int>& y) {
        int n_rows = static_cast<int>(costMatrix.rows());
        int n_cols = static_cast<int>(costMatrix.cols());
        
        x.clear();
        y.clear();
        x.resize(n_rows);
        y.resize(n_cols);
        
        const int n = n_rows + n_cols;
        costLimit /= 2.0;
        
        // Allocate cost matrix
        double **cost_ptr = new double*[n];
        for (int i = 0; i < n; ++i) {
            cost_ptr[i] = new double[n];
            for (int j = 0; j < n; ++j) {
                if (i < n_rows && j < n_cols) {
                    cost_ptr[i][j] = costMatrix(i, j);
                } else if (i >= n_rows && j >= n_cols) {
                    cost_ptr[i][j] = 0.0;
                } else {
                    cost_ptr[i][j] = costLimit;
                }
            }
        }
        
        std::vector<lapjv_internal::int_t> x_c(n), y_c(n);
        lapjv_internal::lapjv_internal(static_cast<lapjv_internal::uint_t>(n), 
                                       cost_ptr, x_c.data(), y_c.data());
        
        // Cleanup
        for (int i = 0; i < n; ++i) delete[] cost_ptr[i];
        delete[] cost_ptr;
        
        // Map results back
        for (int i = 0; i < n_rows; ++i) {
            x[i] = (x_c[i] >= n_cols) ? -1 : static_cast<int>(x_c[i]);
        }
        for (int i = 0; i < n_cols; ++i) {
            y[i] = (y_c[i] >= n_rows) ? -1 : static_cast<int>(y_c[i]);
        }
    }
};

// Convenience function (backward compatibility)
inline void linearAssignment(const Eigen::MatrixXd& costMatrix, double thresh,
                             std::vector<std::vector<int>>& matches,
                             std::vector<int>& unmatched_a,
                             std::vector<int>& unmatched_b) {
    LAPSolver::linearAssignment(costMatrix, thresh, matches, unmatched_a, unmatched_b);
}

} // namespace association
} // namespace trackers

#endif // TRACKERS_ASSOCIATION_LAP_SOLVER_HPP

