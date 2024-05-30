#ifndef CVWT_FILTERBANK_HPP
#define CVWT_FILTERBANK_HPP

#include <opencv2/core.hpp>
#include <string>
#include <memory>
#include <vector>
#include <array>
#include "cvwt/exception.hpp"

namespace cvwt
{
/**
 * @brief A pair of lowpass and highpass filter kernels.
 */
class KernelPair
{
public:
    /**
     * @brief Construct a pair of empty kernels.
     */
    KernelPair() :
        _lowpass(0, 0, CV_64F),
        _highpass(0, 0, CV_64F)
    {}

    /**
     * @brief Construct a pair of kernels.
     *
     * The two kernels must be the same size and type.
     *
     * @param[in] lowpass
     * @param[in] highpass
     */
    KernelPair(const cv::Mat& lowpass, const cv::Mat& highpass) :
        _lowpass(lowpass),
        _highpass(highpass)
    {
        if (lowpass.size() != highpass.size()) {
            internal::throw_bad_size(
                "Kernels must be the same size. ",
                "Got lowpass.size() = ", lowpass.size(),
                " and highpass.size() = ", highpass.size(), "."
            );
        }
        if (lowpass.type() != highpass.type()) {
            internal::throw_bad_arg(
                "Kernels must be the same type. ",
                "Got lowpass.type() = ", internal::get_type_name(lowpass.type()),
                " and highpass.type() = ", internal::get_type_name(highpass.type()), "."
            );
        }
    }
    /**
     * @brief The lowpass kernel coefficients.
     */
    cv::Mat lowpass() const { return _lowpass; }
    /**
     * @brief The highpass kernel coefficients.
     */
    cv::Mat highpass() const { return _highpass; }
    /**
     * @brief Returns true if both kernels are empty.
     */
    bool empty() const { return _lowpass.empty(); }
    /**
     * @see matrix_equals()
     */
    bool operator==(const KernelPair& other) const;

private:
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
};
} // namespace internal




/**
 * @brief Two dimensional discrete wavelet transform filter bank.
 *
 * This class is used as a building block to implement two dimensional multiscale
 * discrete wavelet transforms.
 * It provides both the decomposition (i.e. forward/analysis) and reconstruction
 * (i.e. inverse/synthesis) transformations at a single spatial scale.
 *
 * Decomposition is a cascade of two stages where each
 * stage consists of a lowpass filter \f$g_d[n]\f$ and a highpass
 * filter \f$h_d[n]\f$, each of which is followed by downsampling by
 * two.
 * The first stage convolves along rows and downsamples columns.
 * The second stage convolves along columns and downsamples rows.
 * This results in four outputs:
 * - Approximation Subband Coefficients (Lowpass-Lowpass)
 * - Horizontal Detail Subband Coefficients (Lowpass-Highpass)
 * - Vertical Detail Subband Coefficients (Highpass-Lowpass)
 * - Diagonal Detail Subband Coefficients (Highpass-Highpass)
 * @image html decompose_filter_bank.png "Decomposition Block Diagram"
 *
 * Reconstruction revserses the signal flow, uses the reconstruction
 * kernels \f$g_r[n]\f$ and \f$h_r[n]\f$, and upsamples
 * instead of downsampling.
 * @image html reconstruct_filter_bank.png "Reconstruction Block Diagram"
 */
class FilterBank
{
public:
    /**
     * @brief Construct an empty filter bank.
     */
    FilterBank();
    /**
     * @brief Construct a new filter bank.
     *
     * @param[in] decompose_lowpass
     * @param[in] decompose_highpass
     * @param[in] reconstruct_lowpass
     * @param[in] reconstruct_highpass
     */
    FilterBank(
        const cv::Mat& decompose_lowpass,
        const cv::Mat& decompose_highpass,
        const cv::Mat& reconstruct_lowpass,
        const cv::Mat& reconstruct_highpass
    );
    /**
     * @brief Copy Constructor.
     */
    FilterBank(const FilterBank& other) = default;
    /**
     * @brief Move Constructor.
     */
    FilterBank(FilterBank&& other) = default;

    /**
     * @brief Returns true if the filter kernels are empty.
     *
     * @see cv::Mat::empty()
     */
    bool empty() const { return _p->decompose.empty(); }
    /**
     * @brief Returns the filter kernels data type depth.
     *
     * @see cv::Mat::depth()
     */
    int depth() const { return _p->decompose.depth(); }
    /**
     * @brief Returns maximum number of kernel coefficients.
     *
     * This is equal to `std::max(decompose_kernels.filter_length(), reconstruct_kernels.filter_length()))`.
     */
    int filter_length() const { return _p->filter_length; }
    /**
     * @brief The decomposition kernels.
     */
    KernelPair decompose_kernels() const { return _p->decompose_kernels(); }
    /**
     * @brief The reconstruction kernels.
     */
    KernelPair reconstruct_kernels() const { return _p->reconstruct_kernels(); }

    /**
     * @brief Two filter banks are equal if their decompose_kernels() are equal and their reconstruct_kernels() are equal.
     */
    bool operator==(const FilterBank& other) const;
    friend std::ostream& operator<<(std::ostream& stream, const FilterBank& filter_bank);

    /**
     * @brief Decompose an image.
     *
     * The outputs will all have the same number of channels as the input image
     * and depth equal to `max(image.depth(), depth())`.
     *
     * The size of each output will be equal to subband_size().
     * Because full convolution requires extrapolating the image by
     * the filter_length() - 1 on all sides, the size of the outputs will
     * generally be larger than half the `image.size()`.
     *
     * @param[in] image The image to decompose. This can be any type, single channel or multichannel.
     * @param[out] approx The approximation subband coefficients.
     * @param[out] horizontal_detail The horizontal detail subband coefficients.
     * @param[out] vertical_detail The vertical detail subband coefficients.
     * @param[out] diagonal_detail The diagonal detail subband coefficients.
     * @param[in] border_type The border extrapolation method.
     * @param[in] border_value The border extrapolation value if `border_type` is cv::BORDER_CONSTANT.
     */
    void decompose(
        cv::InputArray image,
        cv::OutputArray approx,
        cv::OutputArray horizontal_detail,
        cv::OutputArray vertical_detail,
        cv::OutputArray diagonal_detail,
        int border_type=cv::BORDER_DEFAULT,
        const cv::Scalar& border_value=cv::Scalar()
    ) const;

    /**
     * @brief Reconstruct an image.
     *
     * The coefficients `approx`, `horizontal_detail`, `vertical_detail`,
     * and `diagonal_detail` must all be the same size, same depth, and have the
     * same number of channels.  If not, an execption is thrown.
     *
     * The output will have the same number of channels as the coefficients
     * and depth equal to `max(approx.depth(), depth())`.
     *
     * The size of the reconstructed image must be provided explicitly via the
     * `output_size` argument so that the extrapolated strip along the border
     * added by decompose() can be discarded.
     * The reconstructed image size cannot be computed from the size of the
     * coefficients because of the unknown integer truncation that occurs when
     * downsampling in decompose().
     *
     * @param[in] approx The approximation subband coefficients.
     * @param[in] horizontal_detail The horizontal detail subband coefficients.
     * @param[in] vertical_detail The vertical detail subband coefficients.
     * @param[in] diagonal_detail The diagonal detail subband coefficients.
     * @param[out] image The reconstructed image.
     * @param[in] image_size The size of the reconstructed image.
     *                        This must be the size of the image passed to
     *                        decompose().
     */
    void reconstruct(
        cv::InputArray approx,
        cv::InputArray horizontal_detail,
        cv::InputArray vertical_detail,
        cv::InputArray diagonal_detail,
        cv::OutputArray image,
        const cv::Size& image_size
    ) const;

    /**
     * @brief Returns the size of each coefficient subband for the given image size.
     *
     * @param[in] image_size The size of the image.
     */
    cv::Size subband_size(const cv::Size& image_size) const;

    /**
     * @brief Swaps and flips the decomposition and reconstruction kernels.
     */
    [[nodiscard]]
    FilterBank reverse() const;

    //  Filter Bank Factories
    /**@{*/
    /**
     * @brief Creates an orthogonal wavelet filter bank.
     *
     * @param[in] reconstruct_lowpass_coeffs
     */
    static FilterBank create_orthogonal_filter_bank(
        cv::InputArray reconstruct_lowpass_coeffs
    );

    /**
     * @brief Creates a biorthogonal wavelet filter bank.
     *
     * @param[in] reconstruct_lowpass_coeffs
     * @param[in] decompose_lowpass_coeffs
     */
    static FilterBank create_biorthogonal_filter_bank(
        cv::InputArray reconstruct_lowpass_coeffs,
        cv::InputArray decompose_lowpass_coeffs
    );
    /**@}*/

    /**
     * @private
     */
    int promote_type(int type) const;
protected:
    void extrapolate_border(
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

private:
    std::shared_ptr<internal::FilterBankImpl> _p;
};

/**
 * @brief Writes a string representation of a FilterBank to an output stream.
 */
std::ostream& operator<<(std::ostream& stream, const FilterBank& filter_bank);
} // namespace cvwt

#endif  // CVWT_FILTERBANK_HPP

