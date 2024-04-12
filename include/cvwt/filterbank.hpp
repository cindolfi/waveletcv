#ifndef CVWT_FILTERBANK_HPP
#define CVWT_FILTERBANK_HPP

#include <opencv2/core.hpp>
#include <string>
#include <memory>
#include <vector>
#include <array>

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
     * @param lowpass
     * @param highpass
     */
    KernelPair(const cv::Mat& lowpass, const cv::Mat& highpass) :
        _lowpass(lowpass),
        _highpass(highpass)
    {
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
     * @brief The number of coefficients in each kernel.
     */
    int filter_length() const { return _lowpass.total(); }
    /**
     * @see cv::Mat::empty()
     */
    bool empty() const { return _lowpass.empty(); }

    /**
     * @see equals()
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


/**
 * @brief Two dimensional discrete wavelet transform filter bank
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
     * @brief Construct an empty Filter Bank object.
     */
    FilterBank();
    /**
     * @brief Construct a new Filter Bank object.
     *
     * @param reconstruct_lowpass
     * @param reconstruct_highpass
     * @param decompose_lowpass
     * @param decompose_highpass
     */
    FilterBank(
        const cv::Mat& reconstruct_lowpass,
        const cv::Mat& reconstruct_highpass,
        const cv::Mat& decompose_lowpass,
        const cv::Mat& decompose_highpass
    );
    /**
     * @brief Construct a new Filter Bank object.
     *
     * @param reconstruct_kernels
     * @param decompose_kernels
     */
    FilterBank(
        const KernelPair& reconstruct_kernels,
        const KernelPair& decompose_kernels
    );
    /**
     * @brief Copy Constructor
     */
    FilterBank(const FilterBank& other) = default;
    /**
     * @brief Move Constructor
     */
    FilterBank(FilterBank&& other) = default;

    /**
     * @brief Returns true if the filter kernels are empty.
     *
     * @see cv::Mat::empty()
     */
    bool empty() const { return _p->decompose.empty(); }
    /**
     * @see cv::Mat::type()
     */
    int type() const { return _p->decompose.type(); }
    /**
     * @see cv::Mat::depth()
     */
    int depth() const { return _p->decompose.depth(); }
    /**
     * @brief Returns maximum number of kernel coefficients.
     *
     * This is equal to `std::max(decompose_kernels.filter_length(), reconstruct_kernels.filter_length()))`.
     *
     * @return int
     */
    int filter_length() const { return _p->filter_length; }
    /**
     * @brief The reconstruction kernels.
     *
     * @return KernelPair
     */
    KernelPair reconstruct_kernels() const { return _p->reconstruct_kernels(); }
    /**
     * @brief The decomposition kernels.
     *
     * @return KernelPair
     */
    KernelPair decompose_kernels() const { return _p->decompose_kernels(); }

    /**
     * @brief
     *
     * @param other
     */
    bool operator==(const FilterBank& other) const;
    friend std::ostream& operator<<(std::ostream& stream, const FilterBank& filter_bank);

    /**
     * @brief Decompose an image
     *
     * The outputs will all have the same number of channels as the input image
     * and depth equal to `max(image.depth(), depth())`.
     *
     * The size of each output will be equal to subband_size().
     * Because full convolution requires extrapolating the image by
     * half the filter_length() on all sides, the size of the outputs will
     * generally be larger than half the `image.size()`.
     *
     * Whenever `image.channels() > 1` or `image.depth() > depth()`
     * the filter kernels must be converted internally to `image.type()`.
     * If this function is going to be called repeatedly on inputs of the
     * same cv::Mat::type(), say in the loop of a multiscale algorithm, the loop
     * should be sandwiched by prepare_decompose() and finish_decompose().
     * This will ensure that the filter kernels are only converted once per loop
     * rather than once per iteration, thereby avoiding the extra overhead.
     * @code{cpp}
     * void some_multiscale_algorithm(const cv::Mat& image, const FilterBank& filter_bank) {
     *     // Prepare to call decompose repeatedly, avoiding potential repeated conversion of the filter kernels.
     *     filter_bank.prepare_decompose(image.type());
     *     while (!done) {
     *         // Repeated calls to decompose() using inputs with types equal to image.type().
     *         filter_bank.decompose(...);
     *      }
     *     // Clean up temporaries.
     *     filter_bank.finish_decomose();
     * @endcode
     * Do **not** put prepare_decompose() and finish_decompose() around a
     * **single isolated** call.
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
     * @brief Prepare for a block of decompose() calls.
     *
     * @see decompose()
     *
     * @param type The type of the image to be decomposed.
     */
    void prepare_decompose(int type) const;
    /**
     * @brief Finish a block of decompose() calls.
     *
     * @see decompose()
     */
    void finish_decompose() const;
    /**
     * @brief Returns true if prepared for a block of decompose() calls.
     *
     * @see decompose()
     *
     * @param type The type of the image to be decomposed.
     */
    bool is_decompose_prepared(int type) const
    {
        return !_p->promoted_decompose.empty()
            && _p->promoted_decompose.type() == promote_type(type);
    }

    /**
     * @brief Reconstruct an image
     *
     * The coefficients `approx`, `horizontal_detail`, `vertical_detail`,
     * and `diagonal_detail` must all be the same size, same depth, and have the
     * same number of channels.  If not, an execption is thrown.
     *
     * The output will have the same number of channels as the coefficients
     * and depth equal to `max(approx.depth(), depth())`.
     *
     * The size of the reconstructed image must be provided explicitly via the
     * `output_size` argument.
     * This is because the extrapolated strip along the border added by
     * decompose() must be discarded.
     * But, the truncation that occurs when odd image sizes are halved during
     * decomposition prevents computing the size of the reconstructed image.
     *
     * Whenever `approx.channels() > 1` or `approx.depth() > depth()`
     * the filter kernels must be converted internally to `approx.type()`.
     * If this function is going to be called repeatedly on inputs of the
     * same cv::Mat::type(), say in the loop of a multiscale algorithm, the loop
     * should be sandwiched by prepare_reconstruct() and finish_reconstruct().
     * This will ensure that the filter kernels are only converted once per loop
     * rather than once per iteration, thereby avoiding the extra overhead.
     * @code{cpp}
     * void some_multiscale_algorithm(const FilterBank& filter_bank, const cv::Mat& approx, ...) {
     *     // Prepare to call reconstruct repeatedly, avoiding potential repeated conversion of the filter kernels.
     *     filter_bank.prepare_reconstruct(approx.type());
     *     while (!done) {
     *         // Repeated calls to reconstruct() using inputs with types equal to image.type().
     *         filter_bank.reconstruct(...);
     *      }
     *     // Clean up temporaries.
     *     filter_bank.finish_reconstruct();
     * @endcode
     * Do **not** put prepare_reconstruct() and finish_reconstruct() around a
     * **single isolated** call.
     *
     * @param[in] approx The approximation subband coefficients.
     * @param[in] horizontal_detail The horizontal detail subband coefficients.
     * @param[in] vertical_detail The vertical detail subband coefficients.
     * @param[in] diagonal_detail The diagonal detail subband coefficients.
     * @param[out] output The reconstructed image.
     * @param[in] output_size The size of the reconstructed image.
     *                        This must be the size of the image passed to
     *                        decompose().
     */
    void reconstruct(
        cv::InputArray approx,
        cv::InputArray horizontal_detail,
        cv::InputArray vertical_detail,
        cv::InputArray diagonal_detail,
        cv::OutputArray output,
        const cv::Size& output_size
    ) const;
    /**
     * @brief Prepare for a block of reconstruct() calls.
     *
     * @see reconstruct()
     *
     * @param type The type of the image to be reconstructed.
     */
    void prepare_reconstruct(int type) const;
    /**
     * @brief Finish a block of reconstruct() calls.
     *
     * @see reconstruct()
     */
    void finish_reconstruct() const;
    /**
     * @brief Returns true if prepared for a block of reconstruct() calls.
     *
     * @see reconstruct()
     *
     * @param type The type of the image to be reconstructed.
     */
    bool is_reconstruct_prepared(int type) const
    {
        return !_p->promoted_reconstruct.empty()
            && _p->promoted_reconstruct.type() == promote_type(type);
    }

    /**
     * @brief Returns the size of each coefficient subband for the given image size.
     *
     * @param image_size The size of the image.
     * @return cv::Size
     */
    cv::Size subband_size(const cv::Size& image_size) const;

    //  Kernel Pair Factories
    /**
     * @brief Create a orthogonal decompose kernels object
     *
     * @param reconstruct_lowpass_coeffs
     * @return KernelPair
     */
    static KernelPair create_orthogonal_decompose_kernels(
        cv::InputArray reconstruct_lowpass_coeffs
    );

    /**
     * @brief Create a orthogonal reconstruct kernels object
     *
     * @param reconstruct_lowpass_coeffs
     * @return KernelPair
     */
    static KernelPair create_orthogonal_reconstruct_kernels(
        cv::InputArray reconstruct_lowpass_coeffs
    );
    /**
     * @brief Create a biorthogonal decompose kernels object
     *
     * @param reconstruct_lowpass_coeffs
     * @param decompose_lowpass_coeffs
     * @return KernelPair
     */
    static KernelPair create_biorthogonal_decompose_kernels(
        cv::InputArray reconstruct_lowpass_coeffs,
        cv::InputArray decompose_lowpass_coeffs
    );
    /**
     * @brief Create a biorthogonal reconstruct kernels object
     *
     * @param reconstruct_lowpass_coeffs
     * @param decompose_lowpass_coeffs
     * @return KernelPair
     */
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

private:
    std::shared_ptr<internal::FilterBankImpl> _p;
};

std::ostream& operator<<(std::ostream& stream, const FilterBank& filter_bank);
} // namespace cvwt

#endif  // CVWT_FILTERBANK_HPP

