#ifndef WAVELET_FILTERBANK_HPP
#define WAVELET_FILTERBANK_HPP

#include <opencv2/core.hpp>
#include <string>
#include <memory>
#include <vector>
#include <array>

namespace wavelet
{
class KernelPair
{
public:
    KernelPair() :
        _lowpass(0, 0, CV_64F),
        _highpass(0, 0, CV_64F)
    {}

    KernelPair(const cv::Mat& lowpass, const cv::Mat& highpass) :
        _lowpass(lowpass),
        _highpass(highpass)
    {}

    template <typename T>
    KernelPair(const std::vector<T>& lowpass, const std::vector<T>& highpass) :
        KernelPair(cv::Mat(lowpass, true), cv::Mat(highpass, true))
    {}

    template <typename T, int N>
    KernelPair(const std::array<T, N>& lowpass, const std::array<T, N>& highpass) :
        KernelPair(cv::Mat(lowpass, true), cv::Mat(highpass, true))
    {}

    cv::Mat lowpass() const { return _lowpass; }
    cv::Mat highpass() const { return _highpass; }
    int filter_length() const { return _lowpass.total(); }
    bool empty() const { return _lowpass.empty(); }

    bool operator==(const KernelPair& other) const;

protected:
    cv::Mat _lowpass;
    cv::Mat _highpass;
};


namespace internal
{
cv::Mat as_column_vector(const cv::Mat& vector);

struct DecomposeKernels
{
    DecomposeKernels() :
        lowpass(0, 0, CV_64F),
        highpass(0, 0, CV_64F)
    {}

    DecomposeKernels(const cv::Mat& lowpass, const cv::Mat& highpass) :
        lowpass(as_column_vector(lowpass)),
        highpass(as_column_vector(highpass))
    {}

    DecomposeKernels(const DecomposeKernels& other) = default;
    DecomposeKernels(DecomposeKernels&& other) = default;

    bool empty() const { return lowpass.empty(); }
    int type() const { return lowpass.type(); }
    int depth() const { return lowpass.depth(); }
    void release()
    {
        lowpass.release();
        highpass.release();
    }

    cv::Mat lowpass;
    cv::Mat highpass;
};

struct ReconstructKernels
{
    ReconstructKernels() :
        even_lowpass(0, 0, CV_64F),
        odd_lowpass(0, 0, CV_64F),
        even_highpass(0, 0, CV_64F),
        odd_highpass(0, 0, CV_64F)
    {}

    ReconstructKernels(
        const cv::Mat& even_lowpass,
        const cv::Mat& odd_lowpass,
        const cv::Mat& even_highpass,
        const cv::Mat& odd_highpass
    ) :
        even_lowpass(as_column_vector(even_lowpass)),
        odd_lowpass(as_column_vector(odd_lowpass)),
        even_highpass(as_column_vector(even_highpass)),
        odd_highpass(as_column_vector(odd_highpass))
    {}

    ReconstructKernels(const ReconstructKernels& other) = default;
    ReconstructKernels(ReconstructKernels&& other) = default;

    bool empty() const { return even_lowpass.empty(); }
    int type() const { return even_lowpass.type(); }
    int depth() const { return even_lowpass.depth(); }
    void release()
    {
        even_lowpass.release();
        odd_lowpass.release();
        even_highpass.release();
        odd_highpass.release();
    }

    cv::Mat even_lowpass;
    cv::Mat odd_lowpass;
    cv::Mat even_highpass;
    cv::Mat odd_highpass;
};

struct FilterBankImpl
{
    FilterBankImpl();

    FilterBankImpl(
        const cv::Mat& decompose_lowpass,
        const cv::Mat& decompose_highpass,
        const cv::Mat& reconstruct_lowpass,
        const cv::Mat& reconstruct_highpass
    );

    void split_kernel_into_odd_and_even_parts(
        cv::InputArray kernel,
        cv::OutputArray even_kernel,
        cv::OutputArray odd_kernel
    ) const;
    void merge_even_and_odd_kernels(
        cv::InputArray even_kernel,
        cv::InputArray odd_kernel,
        cv::OutputArray kernel
    ) const;

    KernelPair decompose_kernels() const;
    KernelPair reconstruct_kernels() const;

    //  Argument Checkers - these can be disabled by building with cmake
    //  option CVWT_ARGUMENT_CHECKING = OFF
    #if CVWT_ARGUMENT_CHECKING_ENABLED
    void throw_if_wrong_size(
        const cv::Mat& reconstruct_lowpass,
        const cv::Mat& reconstruct_highpass,
        const cv::Mat& decompose_lowpass,
        const cv::Mat& decompose_highpass
    ) const;
    #else
    void throw_if_wrong_size(
        const cv::Mat& reconstruct_lowpass,
        const cv::Mat& reconstruct_highpass,
        const cv::Mat& decompose_lowpass,
        const cv::Mat& decompose_highpass
    ) const noexcept
    {}
    #endif  // CVWT_ARGUMENT_CHECKING_ENABLED

    int filter_length;
    DecomposeKernels decompose;
    ReconstructKernels reconstruct;
    //  These are used as temporary caches for promoted kernels to avoid
    //  converting kernels every time FilterBank::decompose or
    //  FilterBank::reconstruct is called by multilevel algorithms (e.g. DWT2D).
    //  The caching and freeing is done by the FilterBank::prepare_*() and
    //  FilterBank::finish_*() methods, respectively.
    DecomposeKernels promoted_decompose;
    ReconstructKernels promoted_reconstruct;
};
} // namespace internal


class FilterBank
{
public:
    FilterBank();
    FilterBank(
        const cv::Mat& reconstruct_lowpass,
        const cv::Mat& reconstruct_highpass,
        const cv::Mat& decompose_lowpass,
        const cv::Mat& decompose_highpass
    );
    FilterBank(
        const KernelPair& reconstruct_kernels,
        const KernelPair& decompose_kernels
    );
    FilterBank(const FilterBank& other) = default;
    FilterBank(FilterBank&& other) = default;

    bool empty() const { return _p->decompose.empty(); }
    int type() const { return _p->decompose.type(); }
    int depth() const { return _p->decompose.depth(); }
    int filter_length() const { return _p->filter_length; }
    KernelPair reconstruct_kernels() const { return _p->reconstruct_kernels(); }
    KernelPair decompose_kernels() const { return _p->decompose_kernels(); }

    bool operator==(const FilterBank& other) const;
    friend std::ostream& operator<<(std::ostream& stream, const FilterBank& filter_bank);

    void decompose(
        cv::InputArray image,
        cv::OutputArray approx,
        cv::OutputArray horizontal_detail,
        cv::OutputArray vertical_detail,
        cv::OutputArray diagonal_detail,
        int border_type=cv::BORDER_DEFAULT,
        const cv::Scalar& border_value=cv::Scalar()
    ) const;
    void prepare_decompose(int type) const;
    void finish_decompose() const;
    bool is_decompose_prepared(int type) const
    {
        return !_p->promoted_decompose.empty()
            && _p->promoted_decompose.type() == promote_type(type);
    }

    void reconstruct(
        cv::InputArray approx,
        cv::InputArray horizontal_detail,
        cv::InputArray vertical_detail,
        cv::InputArray diagonal_detail,
        cv::OutputArray output,
        const cv::Size& output_size
    ) const;

    void reconstruct(
        cv::InputArray approx,
        cv::InputArray horizontal_detail,
        cv::InputArray vertical_detail,
        cv::InputArray diagonal_detail,
        cv::OutputArray output
    ) const
    {
        reconstruct(
            approx,
            horizontal_detail,
            vertical_detail,
            diagonal_detail,
            output,
            cv::Size(approx.size() * 2)
        );
    }
    void prepare_reconstruct(int type) const;
    void finish_reconstruct() const;
    bool is_reconstruct_prepared(int type) const
    {
        return !_p->promoted_reconstruct.empty()
            && _p->promoted_reconstruct.type() == promote_type(type);
    }

    cv::Size output_size(const cv::Size& image_size) const;
    int output_size(int image_size) const;
    cv::Size subband_size(const cv::Size& image_size) const;
    int subband_size(int image_size) const;

    //  Kernel Pair Factories
    static KernelPair create_orthogonal_decompose_kernels(
        cv::InputArray reconstruct_lowpass_coeffs
    );
    static KernelPair create_orthogonal_reconstruct_kernels(
        cv::InputArray reconstruct_lowpass_coeffs
    );
    static KernelPair create_biorthogonal_decompose_kernels(
        cv::InputArray reconstruct_lowpass_coeffs,
        cv::InputArray decompose_lowpass_coeffs
    );
    static KernelPair create_biorthogonal_reconstruct_kernels(
        cv::InputArray reconstruct_lowpass_coeffs,
        cv::InputArray decompose_lowpass_coeffs
    );

    int promote_type(int type) const;
protected:
    void promote_image(
        cv::InputArray image,
        cv::OutputArray promoted_image
    ) const;
    void promote_kernel(
        cv::InputArray kernel,
        cv::OutputArray promoted_kernel,
        int type
    ) const;
    void pad(
        cv::InputArray image,
        cv::OutputArray output,
        int border_type=cv::BORDER_DEFAULT,
        const cv::Scalar& border_value=cv::Scalar()
    ) const;
    void convolve_rows_and_downsample_cols(
        cv::InputArray input,
        cv::OutputArray output,
        const cv::Mat& kernel
    ) const;
    void convolve_cols_and_downsample_rows(
        cv::InputArray input,
        cv::OutputArray output,
        const cv::Mat& kernel
    ) const;
    void upsample_cols_and_convolve_rows(
        cv::InputArray input,
        cv::OutputArray output,
        const cv::Mat& even_kernel,
        const cv::Mat& odd_kernel,
        const cv::Size& output_size
    ) const;
    void upsample_rows_and_convolve_cols(
        cv::InputArray input,
        cv::OutputArray output,
        const cv::Mat& even_kernel,
        const cv::Mat& odd_kernel,
        const cv::Size& output_size
    ) const;

    //  Argument Checkers - these can be disabled by building with cmake
    //  option CVWT_ARGUMENT_CHECKING = OFF
    #if CVWT_ARGUMENT_CHECKING_ENABLED
    void throw_if_decompose_image_is_wrong_size(
        cv::InputArray image
    ) const;
    void throw_if_reconstruct_coeffs_are_wrong_size(
        cv::InputArray approx,
        cv::InputArray horizontal_detail,
        cv::InputArray vertical_detail,
        cv::InputArray diagonal_detail
    ) const;
    #else
    void throw_if_decompose_image_is_wrong_size(
        cv::InputArray image
    ) const noexcept
    {}
    void throw_if_reconstruct_coeffs_are_wrong_size(
        cv::InputArray approx,
        cv::InputArray horizontal_detail,
        cv::InputArray vertical_detail,
        cv::InputArray diagonal_detail
    ) const noexcept
    {}
    #endif  // CVWT_ARGUMENT_CHECKING_ENABLED

protected:
    std::shared_ptr<internal::FilterBankImpl> _p;
};

std::ostream& operator<<(std::ostream& stream, const FilterBank& filter_bank);
} // namespace wavelet

#endif  // WAVELET_FILTERBANK_HPP

