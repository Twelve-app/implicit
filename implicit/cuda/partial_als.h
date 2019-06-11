#ifndef IMPLICIT_CUDA_PARTIAL_ALS_H_
#define IMPLICIT_CUDA_PARTIAL_ALS_H_
#include "implicit/cuda/matrix.h"

// Forward ref: don't require the whole cublas definition here
struct cublasContext;

namespace implicit {

struct CudaPartialLeastSquaresSolver {
    explicit CudaPartialLeastSquaresSolver(int factors);
    ~CudaPartialLeastSquaresSolver();

    void least_squares_init(const CudaDenseMatrix & Y, float regularization);

    void least_squares(
                       int first_user,
                       int user_count,
                       const CudaCSRMatrix & Cui,
                       CudaDenseMatrix * X, const CudaDenseMatrix & Y,
                       int cg_steps) const;

    float calculate_loss(
                        int first_user,
                        int user_count,
                        const CudaCSRMatrix & Cui,
                        const CudaDenseMatrix & X,
                        const CudaDenseMatrix & Y,
                        float regularization);

    CudaDenseMatrix YtY;
    cublasContext * blas_handle;
};
}  // namespace implicit
#endif  // IMPLICIT_PARTIAL_CUDA_ALS_H_
