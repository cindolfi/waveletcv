#ifndef CVWT_DWT2D_HPP
#define CVWT_DWT2D_HPP

#include <string>
#include <vector>
#include <memory>
#include <opencv2/core.hpp>
#include "cvwt/wavelet.hpp"
#include "cvwt/utils.hpp"

namespace cvwt
{
/** \addtogroup dwt2d Discrete Wavelet Transform
 *  @{
 */
/**
 * @brief DWT2D detail subband identifier.
 */
enum Subband {
    /** Coefficients computed by cascading the low pass filter into the high pass filter */
    HORIZONTAL = 0,
    /** Coefficients computed by cascading the high pass filter into the low pass filter */
    VERTICAL = 1,
    /** Coefficients computed by cascading the high pass filter into the high pass filter */
    DIAGONAL = 2,
};


namespace internal
{
class Dwt2dCoeffsImpl
{
public:
    Dwt2dCoeffsImpl() :
        coeff_matrix(),
        levels(0),
        image_size(),
        diagonal_subband_rects(),
        wavelet(),
        border_type(cv::BORDER_DEFAULT)
    {
    }

    Dwt2dCoeffsImpl(
        const cv::Mat& coeff_matrix,
        int levels,
        const cv::Size& image_size,
        const std::vector<cv::Rect>& diagonal_subband_rects,
        const Wavelet& wavelet,
        cv::BorderTypes border_type
    ) :
        coeff_matrix(coeff_matrix),
        levels(levels),
        image_size(image_size),
        diagonal_subband_rects(diagonal_subband_rects),
        wavelet(wavelet),
        border_type(border_type)
    {
    }

    Dwt2dCoeffsImpl(
        const cv::Mat& coeff_matrix,
        int levels,
        const cv::Size& image_size,
        const std::vector<cv::Size>& subband_sizes,
        const Wavelet& wavelet,
        cv::BorderTypes border_type
    ) :
        coeff_matrix(coeff_matrix),
        levels(levels),
        image_size(image_size),
        diagonal_subband_rects(),
        wavelet(wavelet),
        border_type(border_type)
    {
        build_diagonal_subband_rects(subband_sizes);
    }

    void build_diagonal_subband_rects(const std::vector<cv::Size>& subband_sizes)
    {
        cv::Point offset(
            coeff_matrix.size().width,
            coeff_matrix.size().height
        );
        diagonal_subband_rects.clear();
        for (const auto& size : subband_sizes) {
            offset.x = offset.x - size.width;
            offset.y = offset.y - size.height;
            diagonal_subband_rects.emplace_back(offset, size);
        }
    }

    bool has_same_metadata(const Dwt2dCoeffsImpl& coeffs_impl) const
    {
        return coeffs_impl.levels == levels
            && coeffs_impl.image_size == image_size
            && coeffs_impl.diagonal_subband_rects == diagonal_subband_rects
            && coeffs_impl.wavelet == coeffs_impl.wavelet
            && coeffs_impl.border_type == border_type;
    }

public:
    cv::Mat coeff_matrix;
    int levels;
    cv::Size image_size;
    std::vector<cv::Rect> diagonal_subband_rects;
    Wavelet wavelet;
    cv::BorderTypes border_type;
};
} // namespace internal



class CoeffsExpr;

/**
 * @brief A two dimensional multiscale discrete wavelet transform.
 *
 * @image html dwt2d.png "Discrete Wavelet Transform Block Diagram"
 *
 * Image decomposition, which is also called analysis or the forward
 * transformation in the literature, is performed by decompose().
 * @code{cpp}
 * cv::Mat image = ...;
 * DWT2D dwt(Wavelet::create("db2"));
 * DWT2D::Coeffs coeffs = dwt.decompose(image);
 * @endcode
 * Alternatively, instances of this class are callable.
 * @code{cpp}
 * DWT2D::Coeffs coeffs = dwt(image);
 * @endcode
 * A third option is the functional interface.
 * @code{cpp}
 * DWT2D::Coeffs coeffs = dwt2d(image, "db2");
 * @endcode
 *
 * Image reconstruction, which is also called synthesis or the inverse
 * transformation in the literature, is accomplished with reconstruct().
 * @code{cpp}
 * cv::Mat reconstructed_image = dwt.reconstruct(coeffs);
 * @endcode
 * Alternatively, the image can be reconstructed using DWT2D::Coeffs::reconstruct().
 * @code{cpp}
 * cv::Mat reconstructed_image = coeffs.reconstruct();
 * @endcode
 * A third option is the functional interface.
 * @code{cpp}
 * cv::Mat reconstructed_image = idwt2d(coeffs, "db2");
 * @endcode
 *
 * @cite AConciseIntroductionToWavelets
 *
 * @see
 *  - FilterBank
 *  - Wavelet
 *  - dwt2d()
 *  - idwt2d()
 */
class DWT2D
{
public:
    /**
     * @brief The result of a multiscale discrete wavelet transformation.
     *
     * This class is a **view** onto a cv::Mat containing the DWT coefficients.
     * The coefficients at each decomposition level are comprised of three
     * submatrices: the horizontal detail subband, the vertical detail subband,
     * and the horizontal detail subband.
     * There is a single submatrix of approximation coefficients stored
     * alongside the coarsest details.
     * Smaller level indices correspond to smaller scales (i.e. higher resolutions).
     * The submatrices are layed out as (a 4-level decomposition is shown for
     * illustration):
     *
     * @copydetails common_four_level_coeff_matrix
     *
     * The regions labeled H0, V0, and D0 are the level 0 (i.e. finest)
     * horizontal, vertical, and diagonal detail subbands, respectively.
     * Likewise, H1, V1, and D1 are the level 1 coefficients, H2, V2, and D2
     * are the level 2 coefficients, and H3, V3, and D3 are the level 3
     * (i.e. coarsest) coefficients.
     * The approximation coefficients are labeled A.
     *
     * DWT2D::Coeffs objects are not constructed directly, but are created by
     * one of the following methods:
     *  - Returned by DWT2D::decompose()
     *  - Cloned using clone() or empty_clone()
     *  - Created DWT2D::create_coeffs() or DWT2D::create_empty_coeffs()
     *
     * The last method is less common and is only used when algorithmically
     * generating the coefficients (as opposed to computing them using a DWT2D).
     * The only caveat is that default constructed DWT2D::Coeffs can be used as
     * DWT2D output parameters.
     * @code{cpp}
     * cv::Mat image = ...;
     * DWT2D::Coeffs coeffs;
     * // Coefficients are passed as an output parameter.
     * dwt2d(image, coeffs);
     * @endcode
     * But, this can be accompished with
     * @code{cpp}
     * cv::Mat image = ...;
     * // Coefficients are returned.
     * DWT2D::Coeffs coeffs = dwt2d(image);
     * @endcode
     * The performance should be comparable because DWT2D::Coeffs are moveable.
     * The difference is a matter of style.  The first method is more
     * idiomatic of OpenCV code while the second is a bit more succinct.
     *
     * DWT2D::Coeffs objects are implicitly castable to cv::Mat, cv::InputArray,
     * cv::OutputArray, and cv::InputOutputArray.  In each case, the private
     * cv::Mat member is returned.  Since cv::Mat objects share the underlying
     * data, any modification of the casted value will result in modification
     * of the coefficients.  **This means that DWT2D::Coeffs can be passed to
     * any OpenCV function and acted on like a normal cv::Mat.**
     *
     * In addition to the coefficients matrix, a DWT2D::Coeffs object contains
     * metadata defining the structure of subband submatrices and DWT2D object.
     * In cases where an empty DWT2D::Coeffs object is needed it should be
     * created from an existing DWT2D::Coeffs with empty_clone() so that the
     * metadata is retained.
     * For example, consider taking the log of a DWT2D::Coeffs.
     * @code{cpp}
     * DWT2D::Coeffs coeffs = ...;
     * auto log_coeffs = coeffs.empty_clone();
     * cv::log(coeffs, log_coeffs);
     * @endcode
     *
     * In some cases it may be necessary to expicitly cast to a cv::Mat,
     * process the matrix, and then assign the result to a DWT2D::Coeffs.
     * A typical situation involves using matrix operations (i.e +, -, etc.).
     * @code{cpp}
     * DWT2D::Coeffs coeffs = ...;
     * // Must cast to a cv::Mat to use matrix operations.
     * cv::Mat matrix = 2 * static_cast<cv::Mat>(coeffs);
     * @endcode
     * In such cases the matrix can be assigned to an empty clone of the
     * original DWT2D::Coeffs
     * @code{cpp}
     * // Using empty_clone() ensures metadata is retained.
     * auto new_coeffs = coeffs.empty_clone();
     * new_coeffs = matrix;
     * @endcode
     * or they can be assigned to original DWT2D::Coeffs
     * @code{cpp}
     * coeffs = matrix;
     * @endcode
     *
     *
     * @code{cpp}
     * DWT2D::Coeffs coeffs = ...;
     * // Get the diagonal subband detail coefficients at progressively coarser scales
     * for (auto& level_coeffs : coeffs)
     *     level_coeffs.diagonal_detail();
     * @endcode
     */
    class Coeffs
    {
        friend class DWT2D;
        /**
         * @class common_scalar_definition
         *
         * Scalars are defined to be:
         *  - A fundamental type (e.g. float, double, etc.)
         *  - A vector scalar containing channels() elements (e.g. cv::Vec,
         *    std::vector, array, etc.)
         *  - A cv::Scalar if channels() is less than or equal to 4
         */
        /**
         * @class common_scalar_definition_list
         *
         *  A fundamental type (e.g. float, double, etc.)
         *  - A vector scalar containing channels() elements (e.g. cv::Vec,
         *    std::vector, array, etc.)
         *  - A cv::Scalar if channels() is less than or equal to 4
         */

        /**
         * @class common_four_level_coeff_matrix
         *
         * <div class="coeffs-layout-table">
         * <table>
         *     <tr>
         *         <td class="A" style="width: 6.25%; height: 6.25%">A</td>
         *         <td class="V3" style="width: 6.25%; height: 6.25%">V3</td>
         *         <td class="V2" rowspan="2" colspan="1" style="width: 12.5%; height: 12.5%">V2</td>
         *         <td class="V1" rowspan="3" colspan="1" style="width: 25%; height: 25%">V1</td>
         *         <td class="V0" rowspan="4" colspan="1" style="width: 50%; height: 50%">V0</td>
         *     </tr>
         *     <tr>
         *         <td class="H3" style="width: 6.25%; height: 6.25%">H3</td>
         *         <td class="D3" style="width: 6.25%; height: 6.25%">D3</td>
         *     </tr>
         *     <tr>
         *         <td class="H2" rowspan="1" colspan="2" style="width: 12.5%; height: 12.5%">H2</td>
         *         <td class="D2" rowspan="1" colspan="1" style="width: 12.5%; height: 12.5%">D2</td>
         *     </tr>
         *     <tr>
         *         <td class="H1" rowspan="1" colspan="3" style="width: 25%; height: 25%">H1</td>
         *         <td class="D1"  rowspan="1" colspan="1" style="width: 25%; height: 25%">D1</td>
         *     </tr>
         *     <tr>
         *         <td class="H0"  rowspan="1" colspan="4" style="width: 50%; height: 50%">H0</td>
         *         <td class="D0" rowspan="1" colspan="1" style="width: 50%; height: 50%">D0</td>
         *     </tr>
         * </table>
         * </div>
         */

        /**
         * @class common_invalid_coeffs_definition
         *
         * Invalid detail coefficients are half rows or columns of zeros that
         * result from odd sized detail rects.  These are simply internal padding
         * and are not the result of an image decomposition and are not used
         * during reconstruction.
         */
    public:
        class LevelIterator
        {
        public:
            using value_type = Coeffs;
            using difference_type = int;

            LevelIterator() = default;
            LevelIterator(Coeffs* coeffs, int level) : coeffs(coeffs), level(level) {}
            value_type operator*() const { return coeffs->at_level(level); }
            auto& operator++(){ ++level; return *this; }
            auto operator++(int) { auto copy = *this; ++*this; return copy; }
            auto& operator--() { --level; return *this; }
            auto operator--(int) { auto copy = *this; --*this; return copy; }
            bool operator==(const LevelIterator& rhs) const { return coeffs == rhs.coeffs && level == rhs.level; }
            difference_type operator-(const LevelIterator& rhs) const { return level - rhs.level; }
        private:
            Coeffs* coeffs;
            int level;
        };

        class ConstLevelIterator
        {
        public:
            using value_type = const Coeffs;
            using difference_type = int;

            ConstLevelIterator() = default;
            ConstLevelIterator(const Coeffs* coeffs, int level) : coeffs(coeffs), level(level) {}
            value_type operator*() const { return coeffs->at_level(level); }
            auto& operator++(){ ++level; return *this; }
            auto operator++(int) { auto copy = *this; ++*this; return copy; }
            auto& operator--() { --level; return *this; }
            auto operator--(int) { auto copy = *this; --*this; return copy; }
            bool operator==(const ConstLevelIterator& other) const { return coeffs == other.coeffs && level == other.level; }
            difference_type operator-(const ConstLevelIterator& rhs) const { return level - rhs.level; }
        private:
            const Coeffs* coeffs;
            int level;
        };

    protected:
        /**
         * @brief Construct a new Coeffs object.
         *
         * @param[in] coeff_matrix The coefficient matrix.
         * @param[in] levels The number of decomposition levels.
         * @param[in] image_size The size of the decomposed/reconstructed image.
         * @param[in] subband_sizes The sizes of each detail subbands.
         * @param[in] wavelet The DWT2D Wavelet used to decompose/reconstruct
         *                    the image.
         * @param[in] border_type The DWT2D method of border extrapolation.
         */
        Coeffs(
            const cv::Mat& coeff_matrix,
            int levels,
            const cv::Size& image_size,
            const std::vector<cv::Size>& subband_sizes,
            const Wavelet& wavelet,
            cv::BorderTypes border_type
        );

        /**
         * @brief Construct a new Coeffs object.
         *
         * @param[in] coeff_matrix The coefficient matrix.
         * @param[in] levels The number of decomposition levels.
         * @param[in] image_size The size of the decomposed/reconstructed image.
         * @param[in] diagonal_subband_rects The @ref cv::Rect "rects" defining
         *                                   each of the diagonal detail
         *                                   submatrices.
         * @param[in] wavelet The DWT2D Wavelet used to decompose/reconstruct
         *                    the image.
         * @param[in] border_type The DWT2D method of border extrapolation.
         */
        Coeffs(
            const cv::Mat& coeff_matrix,
            int levels,
            const cv::Size& image_size,
            const std::vector<cv::Rect>& diagonal_subband_rects,
            const Wavelet& wavelet,
            cv::BorderTypes border_type
        );

        /**
         * @brief Reset the metadata and (re)create the coefficient matrix.
         *
         * @param[in] size The size of the coefficient matrix.
         * @param[in] type The @ref cv::Mat::type() "type" of the coefficient
         *                 matrix element.
         * @param[in] levels The number of decomposition levels.
         * @param[in] image_size The size of the decomposed/reconstructed image.
         * @param[in] subband_sizes The sizes of each detail subbands.
         * @param[in] wavelet The DWT2D Wavelet used to decompose/reconstruct
         *                    the image.
         * @param[in] border_type The DWT2D method of border extrapolation.
         */
        void reset(
            const cv::Size& size,
            int type,
            int levels,
            const cv::Size& image_size,
            const std::vector<cv::Size>& subband_sizes,
            const Wavelet& wavelet,
            cv::BorderTypes border_type
        );

    public:
        /**
         * @brief Construct an empty Coeffs object.
         */
        Coeffs();
        /**
         * @brief Copy Constructor.
         */
        Coeffs(const Coeffs& other) = default;
        /**
         * @brief Move Constructor.
         */
        Coeffs(Coeffs&& other) = default;

        //  Assignment
        /**
         * @brief Copy Assignment.
         *
         * The reference to the underlying data is copied.  After assighment
         * both `this` and @pref{coeffs} will refer to the same data.
         */
        Coeffs& operator=(const Coeffs& coeffs) = default;

        /**
         * @brief Move Assignment.
         */
        Coeffs& operator=(Coeffs&& coeffs) = default;

        /**
         * @brief Assignment from a matrix or scalar
         *
         * @warning Despite being a wrapper around cv::Mat, DWT2D::Coeffs and
         *          cv::Mat have different assigment semantics.  DWT2D::Coeffs
         *          **copies the values** from the matrix whereas cv::Mat copies
         *          the reference to the underlying data.
         *
         * @param[in] coeffs The coefficients.  This must be one of:
         *  - A matrix of size level_size() "level_size(0)" with channels()
         *    number of channels
         *  - @copydetails common_scalar_definition_list
         *
         * @throws cv::Exception If @pref{coeffs} is an incompatible matrix or
         *                       scalar.
         */
        Coeffs& operator=(cv::InputArray coeffs);

        //  Casting
        /**
         * @brief Implicit cast to cv::Mat.
         *
         * All metadata is discarded (e.g. levels(), wavelet(), border_type(),
         * image_size(), etc.).  As such, it is impossible to cast the result of
         * this operation back to a DWT2D::Coeffs object.
         *
         * This operation does **not** copy the underlying data (i.e. it is O(1)).
         * Rather, it returns a copy of the private cv::Mat object,
         * which has a shared pointer to the numeric data.
         *
         * @warning Modifying the elements of the returned cv::Mat
         *          will modify the coefficients stored by this object.
         */
        operator cv::Mat() const { return _p->coeff_matrix; }

        /**
         * @brief Implicit cast to cv::InputArray.
         *
         * This operation is implemented for interoperability with OpenCV functions.
         * Users should never need to use it directly.
         *
         * This operation is equivalent to:
         * @code{cpp}
         * cv::_InputArray(static_cast<cv::Mat>(*this));
         * @endcode
         *
         * @warning Modifying the elements of the returned cv::_InputArray
         *          will modify the coefficients stored by this object.
         *
         * @see `DWT2D::Coeffs::operator cv::Mat()`
         */
        operator cv::_InputArray() const { return _p->coeff_matrix; }

        /**
         * @brief Implicit cast to cv::OutputArray.
         *
         * This operation is implemented for interoperability with OpenCV functions.
         * Users should never need to use it directly.
         *
         * This operation is equivalent to:
         * @code{cpp}
         * cv::_OutputArray(static_cast<cv::Mat>(*this));
         * @endcode
         *
         * @warning Modifying the elements of the returned cv::_OutputArray
         *          will modify the coefficients stored by this object.
         *
         * @see `DWT2D::Coeffs::operator cv::Mat()`
         */
        operator cv::_OutputArray() const { return _p->coeff_matrix; }

        /**
         * @brief Implicit cast to cv::InputOutputArray.
         *
         * This operation is implemented for interoperability with OpenCV functions.
         * Users should never need to use it directly.
         *
         * This operation is equivalent to:
         * @code{cpp}
         * cv::_InputOutputArray(static_cast<cv::Mat>(*this));
         * @endcode
         *
         * @warning Modifying the elements of the returned cv::_InputOutputArray
         *          will modify the coefficients stored by this object.
         *
         * @see `DWT2D::Coeffs::operator cv::Mat()`
         */
        operator cv::_InputOutputArray() const { return _p->coeff_matrix; }

        //  --------------------------------------------------------------------
        //  Copy
        /**
         * @brief Returns a deep copy of the coefficients matrix and metadata.
         */
        Coeffs clone() const;
        /**
         * @brief Returns an empty Coeffs with a deep copy of the metadata.
         */
        Coeffs empty_clone() const;

        /**
         * @brief Returns an Coeffs with a coefficient matrix and a deep copy of the metadata.
         */
        Coeffs clone_and_assign(cv::InputArray coeff_matrix) const;

        //  --------------------------------------------------------------------
        //  Get & Set Sub-coefficients
        /**
         * @brief Returns the coefficients at and above a decomposition level.
         *
         * For illustration, consider a coefficient matrix returned by a four
         * level DWT2D.
         * The result of `at_level(2)` is the shaded submatrix comprised of the
         * approximation coefficients A and the detail subbands H2, V2, D2, H3,
         * V3, and D3.
         *
         * <div class="shade A H2 V2 D2 H3 V3 D3">
         * @copydetails common_four_level_coeff_matrix
         * </div>
         *
         * @param[in] level
         */
        Coeffs at_level(int level) const;

        /**
         * @brief Sets the coefficients at and above a decomposition level.
         *
         * For illustration, consider a coefficient matrix returned by a four
         * level DWT2D and a given matrix `level_coeffs` of size
         * @ref level_size(int) const "level_size(2)".
         * Calling `set_level(2, level_coeffs)` **copies** `level_coeffs` to the
         * shaded submatrix comprised of the approximation coefficients A and
         * the detail subbands H2, V2, D2, H3, V3, and D3.
         *
         * <div class="shade A H2 V2 D2 H3 V3 D3">
         * @copydetails common_four_level_coeff_matrix
         * </div>
         *
         * @param[in] level
         * @param[in] coeffs The coefficients.  This must be one of:
         *  - A matrix of size level_size() "level_size(level)" with channels()
         *    number of channels
         *  - @copydetails common_scalar_definition_list
         *
         * @throws cv::Exception If @pref{coeffs} is an incompatible matrix or
         *                       scalar.
         */
        void set_level(int level, cv::InputArray coeffs);

        //  --------------------------------------------------------------------
        //  Get & Set Approximation Coefficients
        ///@{
        /**
         * @brief Returns the approximation coefficients.
         */
        cv::Mat approx() const
        {
            return _p->coeff_matrix(approx_rect());
        }

        /**
         * @brief Sets the approximation coefficients.
         *
         * @param[in] coeffs The approximation coefficients.  This must be one of:
         *  - A matrix of size approx_size() with channels() number of channels
         *  - @copydetails common_scalar_definition_list
         *
         * @throws cv::Exception If @pref{coeffs} is an incompatible matrix or scalar.
         */
        void set_approx(cv::InputArray coeffs);
        ///@}

        //  --------------------------------------------------------------------
        //  Get & Set Detail Coefficients (via parameter)
        ///@{
        /**
         * @brief Returns the detail coefficients at a given level and subband.
         *
         * @param[in] level
         * @param[in] subband The detail subband.  This must be a #Subband.
         */
        cv::Mat detail(int level, int subband) const;

        /**
         * @brief Returns the smallest scale detail coefficients in a given subband.
         *
         * @param[in] subband The detail subband.  This must be a #Subband.
         */
        cv::Mat detail(int subband) const { return detail(0, subband); }

        /**
         * @brief Sets all detail coefficients.
         *
         * Only the elements indicated by detail_mask() are copied from
         * @pref{coeffs}.
         *
         * @param[in] coeffs The detail coefficients.  This must be one of:
         *  - A matrix of size size() with channels() number of channels
         *  - @copydetails common_scalar_definition_list
         *
         * @throws cv::Exception If @pref{coeffs} is an incompatible matrix or scalar.
         */
        void set_all_detail_levels(cv::InputArray coeffs);

        /**
         * @brief Sets the detail coefficients at a given level and subband.
         *
         * @param[in] level The scale level.
         * @param[in] subband The detail subband.  This must be a #Subband.
         * @param[in] coeffs The detail subband coefficients.  This must be one of:
         *  - A matrix of size @ref detail_size(int) const "detail_size()" with
         *    channels() number of channels
         *  - @copydetails common_scalar_definition_list
         *
         * @throws cv::Exception
         *  - If @pref{coeffs} is an incompatible matrix or scalar.
         *  - If @pref{subband} is not a valid #Subband.
         */
        void set_detail(int level, int subband, cv::InputArray coeffs);

        /**
         * @brief Sets the smallest scale detail coefficients in a given subband.
         *
         * @param[in] subband The detail subband.  This must be a #Subband.
         * @param[in] coeffs The detail subband coefficients.  This must be one of:
         *  - A matrix of size @ref detail_size(int) const "detail_size()" with
         *    channels() number of channels
         *  - @copydetails common_scalar_definition_list
         *
         * @throws cv::Exception
         *  - If @pref{coeffs} is an incompatible matrix or scalar.
         *  - If @pref{subband} is not a valid #Subband.
         */
        void set_detail(int subband, cv::InputArray coeffs) { set_detail(0, subband, coeffs); }

        /**
         * @brief Returns a collection of detail coefficients at each level in a given subband.
         *
         * This is equivalent to:
         * @code{cpp}
         * std::vector<cv::Mat> collected_details(this->levels());
         * for (int level = 0; level < this->levels(); ++level)
         *     collected_details[level] = this->detail(level, subband);
         * @endcode
         *
         * For illustration, consider a coefficient matrix returned by a four
         * level DWT2D.
         * The result of `collect_details(Subband::HORIZONTAL)` is a vector
         * containing the shaded submatrices H0, H1, H2, and H3.
         *
         * <div class="shade H0 H1 H2 H3">
         * @copydetails common_four_level_coeff_matrix
         * </div>
         *
         * @param[in] subband The detail subband.  This must be a #Subband.
         *
         * @see
         *  - collect_horizontal_details()
         *  - collect_vertical_details()
         *  - collect_diagonal_details()
         */
        std::vector<cv::Mat> collect_details(int subband) const;
        ///@}

        //  --------------------------------------------------------------------
        //  Get & Set Horizontal Detail Coefficients
        ///@{
        /**
         * @brief Returns the horizontal subband detail coefficients at a given level.
         *
         * @param[in] level
         */
        cv::Mat horizontal_detail(int level) const
        {
            return _p->coeff_matrix(horizontal_detail_rect(level));
        }

        /**
         * @brief Returns the smallest scale horizontal subband detail coefficients.
         *
         * This overload is useful when iterating over decomposition levels
         * using a range based for loop.
         */
        cv::Mat horizontal_detail() const { return horizontal_detail(0); }

        /**
         * @brief Sets the horizontal subband detail coefficients at a given level.
         *
         * @param[in] level The scale level.
         * @param[in] coeffs The detail coefficients.  This must be one of:
         *  - A matrix of size @ref detail_size(int) const "detail_size()" with
         *    channels() number of channels
         *  - @copydetails common_scalar_definition_list
         *
         * @throws cv::Exception If @pref{coeffs} is an incompatible matrix or
         *                       scalar.
         */
        void set_horizontal_detail(int level, cv::InputArray coeffs);

        /**
         * @brief Sets the smallest scale horizontal subband detail coefficients.
         *
         * @param[in] coeffs The detail coefficients.  This must be one of:
         *  - A matrix of size @ref detail_size(int) const "detail_size()" with
         *    channels() number of channels
         *  - @copydetails common_scalar_definition_list
         *
         * @throws cv::Exception If @pref{coeffs} is an incompatible matrix or
         *                       scalar.
         */
        void set_horizontal_detail(cv::InputArray coeffs) { set_horizontal_detail(0, coeffs); }

        /**
         * @brief Returns a collection of horizontal subband detail coefficients at each level.
         *
         * This is equivalent to
         * @ref collect_details(int) const "collect_details(Subband::HORIZONTAL)".
         */
        std::vector<cv::Mat> collect_horizontal_details() const { return collect_details(HORIZONTAL); }
        ///@}

        //  --------------------------------------------------------------------
        //  Get & Set Vertical Detail Coefficients
        ///@{
        /**
         * @brief Returns the vertical subband detail coefficients at a given level.
         *
         * @copydetails horizontal_detail(int)
         */
        cv::Mat vertical_detail(int level) const
        {
            return _p->coeff_matrix(vertical_detail_rect(level));
        }

        /**
         * @brief Returns the smallest scale vertical subband detail coefficients.
         *
         * @copydetails horizontal_detail()
         */
        cv::Mat vertical_detail() const { return vertical_detail(0); }

        /**
         * @brief Sets the vertical subband detail coefficients at a given level.
         *
         * @copydetails set_horizontal_detail(int, cv::InputArray)
         */
        void set_vertical_detail(int level, cv::InputArray coeffs);

        /**
         * @brief Sets the smallest scale vertical subband detail coefficients.
         *
         * @copydetails set_horizontal_detail(cv::InputArray)
         */
        void set_vertical_detail(cv::InputArray coeffs) { set_vertical_detail(0, coeffs); }

        /**
         * @brief Returns a collection of vertical subband detail coefficients at each level.
         *
         * This is equivalent to
         * @ref collect_details(int) const "collect_details(Subband::VERTICAL)".
         */
        std::vector<cv::Mat> collect_vertical_details() const { return collect_details(VERTICAL); }
        ///@}

        //  --------------------------------------------------------------------
        //  Get & Set Diagonal Detail Coefficients
        ///@{
        /**
         * @brief Returns the diagonal subband detail coefficients at a given level.
         *
         * @copydetails horizontal_detail(int)
         */
        cv::Mat diagonal_detail(int level) const
        {
            return _p->coeff_matrix(diagonal_detail_rect(level));
        }

        /**
         * @brief Returns the smallest scale diagonal subband detail coefficients.
         *
         * @copydetails horizontal_detail()
         */
        cv::Mat diagonal_detail() const { return diagonal_detail(0); }

        /**
         * @brief Sets the diagonal subband detail coefficients at a given level.
         *
         * @copydetails set_horizontal_detail(int, cv::InputArray)
         */
        void set_diagonal_detail(int level, cv::InputArray coeffs);

        /**
         * @brief Sets the smallest scale diagonal subband detail coefficients.
         *
         * @copydetails set_horizontal_detail(cv::InputArray)
         */
        void set_diagonal_detail(cv::InputArray coeffs) { set_diagonal_detail(0, coeffs); }

        /**
         * @brief Returns a collection of diagonal subband detail coefficients at each level.
         *
         * This is equivalent to
         * @ref collect_details(int) const "collect_details(Subband::DIAGONAL)".
         */
        std::vector<cv::Mat> collect_diagonal_details() const { return collect_details(DIAGONAL); }
        ///@}

        //  --------------------------------------------------------------------
        //  Sizes & Rects
        ///@{
        /**
         * @brief The size of the coefficents starting at the given level.
         *
         * The level size will be empty if and only if the image_size() is
         * empty.  This differs from size(), which equals level_size(0) when
         * this is nonempty and empty when this is empty().
         *
         * The difference is that size() measures the size of the coefficients
         * matrix, whereas level_size() is effectively the sum of subband sizes,
         * which are stored as metadata.
         *
         * The distinction only applies to coefficients created by empty_clone()
         * or DWT2D::create_empty_coeffs().
         *
         * @param[in] level
         *
         * @see level_rect()
         */
        cv::Size level_size(int level) const;

        /**
         * @brief The region containing the coefficients starting at the given level.
         *
         * The level rect will be empty if and only if the image_size() is
         * empty.
         *
         * @param[in] level
         *
         * @see level_size()
         */
        cv::Rect level_rect(int level) const;

        /**
         * @brief The size of the approximation coefficients.
         */
        cv::Size approx_size() const { return detail_size(levels() - 1); }

        /**
         * @brief The region containing the approximation coefficients.
         */
        cv::Rect approx_rect() const;

        /**
         * @brief The size of the each of the detail subbands at the given level.
         *
         * @param[in] level
         */
        cv::Size detail_size(int level) const;

        /**
         * @brief The size of the each of the detail subbands at the finest scale.
         */
        cv::Size detail_size() const { return detail_size(0); }

        /**
         * @brief The region containing the coefficients for the given level and subband.
         *
         * @param[in] level
         * @param[in] subband The detail subband.  This must be a #Subband.
         */
        cv::Rect detail_rect(int level, int subband) const;

        /**
         * @brief The region containing the smallest scale coefficients in the given subband.
         *
         * @param[in] subband The detail subband.  This must be a #Subband.
         */
        cv::Rect detail_rect(int subband) const { return detail_rect(0, subband); }

        /**
         * @brief The region containing the horizontal subband coefficients at the given level.
         *
         * @param[in] level
         */
        cv::Rect horizontal_detail_rect(int level) const;
        /**
         * @brief The region containing the horizontal subband coefficients at the finest scale.
         */
        cv::Rect horizontal_detail_rect() const { return horizontal_detail_rect(0); }

        /**
         * @brief The region containing the vertical subband coefficients at the given level.
         *
         * @param[in] level
         */
        cv::Rect vertical_detail_rect(int level) const;
        /**
         * @brief The region containing the vertical subband coefficients at the finest scale.
         */
        cv::Rect vertical_detail_rect() const { return vertical_detail_rect(0); }

        /**
         * @brief The region containing the diagonal subband coefficients at the given level.
         *
         * @param[in] level
         */
        cv::Rect diagonal_detail_rect(int level) const;
        /**
         * @brief The region containing the diagonal subband coefficients at the finest scale.
         */
        cv::Rect diagonal_detail_rect() const { return diagonal_detail_rect(0); }
        ///@}

        //  --------------------------------------------------------------------
        //  Masks
        ///@{
        /**
         * @brief The mask indicating the invalid detail coefficients.
         *
         * @copydetails common_invalid_coeffs_definition
         *
         * Users should typically use detail_mask() when operating on
         * coefficients over one or more levels or subbands.
         */
        cv::Mat invalid_detail_mask() const;

        /**
         * @brief Returns the total number of valid coefficients.
         *
         * This is equal to:
         *  - `total_details() + approx().total()`
         *  - `total() - cv::countNonZero(invalid_detail_mask())`
         *
         * @copydetails common_invalid_coeffs_definition
         *
         * @see total_details()
         */
        int total_valid() const;

        /**
         * @brief Returns the total number of valid detail coefficients.
         *
         * This is equal to `total() - cv::countNonZero(invalid_detail_mask()) - approx().total()`
         *
         * @copydetails common_invalid_coeffs_definition
         *
         * @see total_valid()
         */
        int total_details() const;

        /**
         * @brief The mask indicating the approximation coefficients.
         */
        cv::Mat approx_mask() const;

        /**
         * @brief The mask indicating the detail coefficients.
         */
        cv::Mat detail_mask() const;

        /**
         * @brief The mask indicating the detail coefficients at a level.
         *
         * @param[in] level
         */
        cv::Mat detail_mask(int level) const;

        /**
         * @brief The mask indicating the subband coefficients at a level.
         *
         * @param[in] levels
         * @param[in] subband The detail subband.  This must be a #Subband.
         */
        cv::Mat detail_mask(int level, int subband) const;

        /**
         * @brief The mask indicating the detail coefficients at a range of levels.
         *
         * @param[in] levels
         */
        cv::Mat detail_mask(const cv::Range& levels) const;

        /**
         * @brief The mask indicating the subband coefficients at a range of levels.
         *
         * @param[in] levels
         * @param[in] subband The detail subband.  This must be a #Subband.
         */
        cv::Mat detail_mask(const cv::Range& levels, int subband) const;

        /**
         * @overload
         *
         * @param[in] lower_level
         * @param[in] upper_level
         * @param[in] subband The detail subband.  This must be a #Subband.
         */
        cv::Mat detail_mask(int lower_level, int upper_level, int subband) const;

        /**
         * @brief The mask indicating the horizontal subband coefficients at the given level.
         *
         * @param[in] level
         */
        cv::Mat horizontal_detail_mask(int level) const;
        /**
         * @brief The mask indicating the horizontal subband coefficients at a range of levels.
         *
         * @param[in] levels
         */
        cv::Mat horizontal_detail_mask(const cv::Range& levels) const;
        /**
         * @overload
         *
         * @param[in] lower_level
         * @param[in] upper_level
         */
        cv::Mat horizontal_detail_mask(int lower_level, int upper_level) const;
        /**
         * @brief The mask indicating the horizontal subband coefficients at the finest scale.
         */
        cv::Mat horizontal_detail_mask() const { return horizontal_detail_mask(0); }

        /**
         * @brief The mask indicating the vertical subband coefficients at the given level.
         *
         * @param[in] level
         */
        cv::Mat vertical_detail_mask(int level) const;
        /**
         * @brief The mask indicating the vertical subband coefficients at a range of levels.
         *
         * @param[in] levels
         */
        cv::Mat vertical_detail_mask(const cv::Range& levels) const;
        /**
         * @overload
         *
         * @param[in] lower_level
         * @param[in] upper_level
         */
        cv::Mat vertical_detail_mask(int lower_level, int upper_level) const;
        /**
         * @brief The mask indicating the vertical subband coefficients at the finest scale.
         */
        cv::Mat vertical_detail_mask() const { return vertical_detail_mask(0); }

        /**
         * @brief The mask indicating the diagonal subband coefficients at the given level.
         *
         * @param[in] level
         */
        cv::Mat diagonal_detail_mask(int level) const;
        /**
         * @brief The mask indicating the diagonal subband coefficients at a range of levels.
         *
         * @param[in] levels
         */
        cv::Mat diagonal_detail_mask(const cv::Range& levels) const;
        /**
         * @overload
         *
         * @param[in] lower_level
         * @param[in] upper_level
         */
        cv::Mat diagonal_detail_mask(int lower_level, int upper_level) const;
        /**
         * @brief The mask indicating the diagonal subband coefficients at the finest scale.
         */
        cv::Mat diagonal_detail_mask() const { return diagonal_detail_mask(0); }
        ///@}

        //  --------------------------------------------------------------------
        //  Convenience cv::Mat Wrappers
        ///@{
        /**
         * @brief The number of rows.
         */
        int rows() const { return _p->coeff_matrix.rows; }

        /**
         * @brief The number of columns.
         */
        int cols() const { return _p->coeff_matrix.cols; }

        /**
         * @brief Returns the size of a coefficients matrix.
         */
        cv::Size size() const { return _p->coeff_matrix.size(); }

        /**
         * @brief Returns the type of a coefficient.
         *
         * This is a convenience wrapper around cv::Mat::type().
         */
        int type() const { return _p->coeff_matrix.type(); }

        /**
         * @brief Returns the depth of a coefficient.
         *
         * This is a convenience wrapper around cv::Mat::depth().
         */
        int depth() const { return _p->coeff_matrix.depth(); }

        /**
         * @brief Returns the number of matrix channels.
         *
         * This is a convenience wrapper around cv::Mat::channels().
         */
        int channels() const { return _p->coeff_matrix.channels(); }

        /**
         * @brief Returns true if the coefficients matrix has no elements.
         *
         * This is a convenience wrapper around cv::Mat::empty().
         */
        bool empty() const { return _p->coeff_matrix.empty(); }

        /**
         * @brief Returns the total number of elements in the coefficient matrix.
         *
         * This is a convenience wrapper around cv::Mat::total().
         */
        size_t total() const { return _p->coeff_matrix.total(); }

        /**
         * @brief Returns the matrix element size in bytes.
         *
         * This is a convenience wrapper around cv::Mat::elemSize().
         */
        size_t elemSize() const { return _p->coeff_matrix.elemSize(); }

        /**
         * @brief Returns the size of each matrix element channel in bytes.
         *
         * This is a convenience wrapper around cv::Mat::elemSize1().
         */
        size_t elemSize1() const { return _p->coeff_matrix.elemSize1(); }

        /**
         * @brief Copies the coefficients to another matrix.
         *
         * This is a convenience wrapper around
         * @ref cv::Mat::copyTo(cv::OutputArray) const "cv::Mat::copyTo()".
         *
         * @param[in] destination Destination matrix. If it does not have a
         *                        proper size or type before the operation, it
         *                        is reallocated.
         */
        void copyTo(cv::OutputArray destination) const { _p->coeff_matrix.copyTo(destination); }

        /**
         * @brief Copies the coefficients to another matrix.
         *
         * This is a convenience wrapper around
         * @ref cv::Mat::copyTo(cv::OutputArray, cv::InputArray) const "cv::Mat::copyTo()".
         *
         * @param[in] destination Destination matrix. If it does not have a
         *                        proper size or type before the operation, it
         *                        is reallocated.
         * @param[in] mask Operation mask of the same size as *this. Its
         *                 non-zero elements indicate which matrix elements need
         *                 to be copied. The mask has to be of type CV_8U and
         *                 can have 1 or multiple channels.
         */
        void copyTo(cv::OutputArray destination, cv::InputArray mask) const { _p->coeff_matrix.copyTo(destination, mask); }

        /**
         * @brief Sets all or some of the coefficients to the specified value.
         *
         * This is a convenience wrapper around cv::Mat::setTo().
         *
         * @param value Assigned scalar converted to the type().
         * @param mask Operation mask of the same size as *this. Its non-zero
         *             elements indicate which matrix elements need to be
         *             copied. The mask has to be of type CV_8U and can have 1
         *             or multiple channels.
         */
        void setTo(cv::InputArray value, cv::InputArray mask = cv::noArray()) const { _p->coeff_matrix.setTo(value, mask); }

        /**
         * @brief Converts the coefficients to another data type with optional scaling.
         *
         * This is a convenience wrapper around cv::Mat::convertTo().
         *
         * @param[in] destination The destination matrix.  If it does not have a
         *                        proper size or type before the operation, it
         *                        is reallocated.
         * @param[in] type The destination matrix type or, rather, the depth
         *                 since the destination will have channels() number of
         *                 channels.  If negative, the destination type will be
         *                 type().
         * @param[in] alpha Optional scale factor.
         * @param[in] beta Optional delta added to the scaled values.
         */
        void convertTo(cv::OutputArray destination, int type, double alpha=1.0, double beta=0.0) const { _p->coeff_matrix.convertTo(destination, type, alpha, beta); }

        /**
         * @brief Returns true if the coefficient matrix is stored continuously in memory.
         *
         * This is a convenience wrapper around cv::Mat::isContinuous().
         */
        bool isContinuous() const { return _p->coeff_matrix.isContinuous(); }

        /**
         * @brief Returns true if the coefficient matrix is a submatrix of another matrix.
         *
         * This is a convenience wrapper around cv::Mat::isSubmatrix().
         */
        bool isSubmatrix() const { return _p->coeff_matrix.isSubmatrix(); }

        CoeffsExpr mul(cv::InputArray matrix, double scale = 1.0) const;
        CoeffsExpr mul(const Coeffs& coeffs, double scale = 1.0) const;
        CoeffsExpr mul(const CoeffsExpr& expression, double scale = 1.0) const;
        ///@}

        //  --------------------------------------------------------------------
        //  Level Iterators
        ///@{
        /**
         * @brief Iterator over decomposition levels set to the finest scale.
         *
         * This function returns an iterator over DWT2D::Coeffs objects starting at progressively coarser decomposition scales.
         * It does **not** return an element-wise iterator.
         * @code{cpp}
         * DWT2D::Coeffs coeffs = ...;
         * int level = 0;
         * for (auto& level_coeffs : coeffs) {
         *     assert(level_coeffs == coeffs.at_level(level));
         *     assert(level_coeffs.levels() == coeffs.levels() - level);
         *     assert(level_coeffs.size() == coeffs.level_size(level));
         *     ++level;
         * }
         * @endcode
         * Element-wise iteration is accomplished by casting to a cv::Mat and using cv::Mat::begin(), i.e. `cv::Mat(coeffs)::begin<PixelType>()`.

         * The overloaded member functions that do not take a level argument are useful when iterating using the range based for loop.
         *
         * The following
         * @code{cpp}
         * DWT2D::Coeffs coeffs = ...;
         * for (auto& level_coeffs : coeffs) {
         *     level_coeffs.set_horizontal_details(...);
         *     level_coeffs.horizontal_details();
         *     level_coeffs.detail_size();
         *     // etc
         * }
         * @endcode
         * is equivalent to
         * @code{cpp}
         * DWT2D::Coeffs coeffs = ...;
         * for (int level = 0; level < coeffs.levels(); ++level) {
         *     coeffs.set_horizontal_details(level, ...);
         *     coeffs.horizontal_details(level);
         *     coeffs.detail_size(level);
         *     // etc
         * }
         * @endcode
         *
         * @see at_level()
         */
        LevelIterator begin() { return LevelIterator(this, 0); }
        ConstLevelIterator begin() const { return ConstLevelIterator(this, 0); }

        /**
         * @brief Iterator over decomposition levels set after the coarsest scale.
         *
         * @see begin()
         */
        LevelIterator end() { return LevelIterator(this, levels()); }
        ConstLevelIterator end() const { return ConstLevelIterator(this, levels()); }

        /**
         * @brief Constant iterator over decomposition levels set to the finest scale.
         *
         * @see begin(), end()
         */
        ConstLevelIterator cbegin() const { return ConstLevelIterator(this, 0); }
        ConstLevelIterator cbegin() { return ConstLevelIterator(this, 0); }

        /**
         * @brief Constant iterator over decomposition levels set after the coarsest scale.
         *
         * @see begin(), end()
         */
        ConstLevelIterator cend() const { return ConstLevelIterator(this, levels()); }
        ConstLevelIterator cend() { return ConstLevelIterator(this, levels()); }
        ///@}

        //  --------------------------------------------------------------------
        //  DWT
        ///@{
        /**
         * @brief Returns the number of decomposition levels.
         */
        int levels() const { return _p->levels; }

        /**
         * @brief Returns the wavelet used to generate the coefficients.
         */
        Wavelet wavelet() const { return _p->wavelet; }

        /**
         * @brief Returns the border exptrapolation method used during decomposition.
         */
        cv::BorderTypes border_type() const { return _p->border_type; }

        /**
         * @brief Returns the DWT2D transformation object used to compute the coefficients.
         */
        DWT2D dwt() const;

        /**
         * @brief Returns the size of the image reconstructed from the coefficients at the given level.
         *
         * @param[in] level
         */
        cv::Size image_size(int level) const { return level == 0 ? _p->image_size : diagonal_detail_rect(level - 1).size(); }

        /**
         * @brief Returns the size of the image reconstructed from the coefficients.
         */
        cv::Size image_size() const { return image_size(0); }

        /**
         * @brief Transform from DWT space back to image space.
         *
         * @see DWT2D::reconstruct()
         */
        cv::Mat reconstruct() const;

        /**
         * @overload
         *
         * @param[out] image The reconstructed image.
         *
         * @see DWT2D::reconstruct()
         */
        void reconstruct(cv::OutputArray image) const;
        ///@}

        //  --------------------------------------------------------------------
        //  Other
        ///@{
        /**
         * @brief Scales and shifts detail coefficients to [0, 1].
         *
         * This function maps detail coefficients centered at 0.5 to detail
         * coefficients centered at 0.
         *
         * The normalized coefficients \f$\tilde\w\f$ are
         * \f{equation}{
         *     \tilde\w = \alpha w + \frac{1}{2}
         * \f}
         * where
         * \f{equation}{
         *     \alpha = \frac{1}{2 \max(|w|)}
         * \f}
         *
         * @note This function is useful for displaying and saving coefficients
         *       as a normal image.  Only the detail coefficients are transformed.
         *       The approximation coefficients are left unchanged, thereby
         *       changing the relative scale between the approximation and
         *       detail coefficients. Reconstruction from the normalized
         *       coefficients will result in distortion.
         *
         * @see
         *  - map_details_from_unit_interval()
         *  - map_detail_to_unit_interval_scale()
         *
         * @param[in] read_mask Indicates which coefficients are used to compute the
         *                  map parameters. This can be a single channel or
         *                  multichannel matrix with depth CV_8U.
         * @param[in] write_mask Indicates which coefficients are mapped.
         *                   This can be a single channel or multichannel matrix
         *                   with depth CV_8U.
         */
        [[nodiscard]]
        DWT2D::Coeffs map_details_to_unit_interval(
            cv::InputArray read_mask = cv::noArray(),
            cv::InputArray write_mask = cv::noArray()
        ) const;

        /**
         * @overload
         *
         * @param[out] scale_output The computed scaling parameter used to map
         *                          the detail coefficients into the interval
         *                          [0, 1].
         * @param[in] read_mask Indicates which coefficients are used to compute the
         *                      map parameters. This can be a single channel or
         *                      multichannel matrix with depth CV_8U.
         * @param[in] write_mask Indicates which coefficients are mapped.
         *                       This can be a single channel or multichannel
         *                       matrix with depth CV_8U.
         */
        [[nodiscard]]
        DWT2D::Coeffs map_details_to_unit_interval(
            double& scale_output,
            cv::InputArray read_mask = cv::noArray(),
            cv::InputArray write_mask = cv::noArray()
        ) const;

        // /**
        //  * @brief Scales and shifts detail coefficients to [0, 1].
        //  *
        //  * This function maps detail coefficients centered at 0.5 to detail
        //  * coefficients centered at 0.
        //  *
        //  * The normalized coefficients \f$\tilde\w\f$ are
        //  * \f{equation}{
        //  *     \tilde\w = \alpha w + \frac{1}{2}
        //  * \f}
        //  * where
        //  * \f{equation}{
        //  *     \alpha = \frac{1}{2 \max(|w|)}
        //  * \f}
        //  *
        //  * @note This function is useful for displaying and saving coefficients
        //  *       as a normal image.  Only the detail coefficients are transformed.
        //  *       The approximation coefficients are left unchanged, thereby
        //  *       changing the relative scale between the approximation and
        //  *       detail coefficients. Reconstruction from the normalized
        //  *       coefficients will result in distortion.
        //  *
        //  * @see
        //  *  - map_details_from_unit_interval()
        //  *  - map_detail_to_unit_interval_scale()
        //  *
        //  * @param[out] normalized_coeffs
        //  * @param[in] read_mask Indicates which coefficients are used to compute the
        //  *                      map parameters. This can be a single channel or
        //  *                      multichannel matrix with depth CV_8U.
        //  * @param[in] write_mask Indicates which coefficients are mapped.
        //  *                       This can be a single channel or multichannel matrix
        //  *                       with depth CV_8U.
        //  */
        // double map_details_to_unit_interval(
        //     Coeffs& normalized_coeffs,
        //     cv::InputArray read_mask = cv::noArray(),
        //     cv::InputArray write_mask = cv::noArray()
        // ) const;

        /**
         * @brief Scales and shifts detail coefficients from [0, 1].
         *
         * This function maps detail coefficients centered at 0.5 to detail
         * coefficients centered at 0.
         *
         * Given the scale parameter \f$\alpha\f$ and the normalized coefficients
         * \f$\tilde\w\f$, this function computes the coefficents \f$w\f$ by
         * \f{equation}{
         *     w = \frac{\tilde\w - \frac{1}{2}}{\alpha}
         * \f}
         *
         * For a particular \f$\max(|w|\f$, the scale parameter \f$\alpha\f$ must be
         * \f{equation}{
         *     \alpha = \frac{1}{2 \max(|w|)}
         * \f}
         *
         * This is the inverse to map_details_to_unit_interval().  It must be
         * called with the scale returned by map_details_to_unit_interval() and
         * the same write mask that was passed to map_details_to_unit_interval().
         *
         * @code{cpp}
         * DWT2D::Coeffs coeffs = ...;
         *
         * // Map the details to [0, 1] such that 0 gets mapped to 0.5
         * DWT2D::Coeffs unit_interval_detail_coeffs;
         * double scale = coeffs.map_details_to_unit_interval(unit_interval_detail_coeffs);
         *
         * // Invert the mapping, i.e. coeffs2 == coeffs element-wise
         * auto coeffs2 = unit_interval_detail_coeffs.map_details_from_unit_interval(scale);
         * @endcode
         *
         * @see
         *  - map_details_to_unit_interval()
         *  - map_detail_to_unit_interval_scale()
         *
         * @param[in] write_mask Indicates which coefficients are mapped.
         *                       This can be a single channel or multichannel
         *                       matrix with depth CV_8U.
         */
        [[nodiscard]]
        DWT2D::Coeffs map_details_from_unit_interval(
            double scale,
            cv::InputArray write_mask = cv::noArray()
        ) const;

        /**
         * @brief Returns the scaling parameter used to map the detail
         *        coefficients into the interval [0, 1].
         *
         * \f{equation}{
         *     \alpha = \frac{1}{2 \max(|w|)}
         * \f}
         *
         * @param[in] read_mask Indicates which coefficients are used to compute
         *                      the scale parameter. This can be a single
         *                      channel or multichannel matrix with depth CV_8U.
         */
        double map_detail_to_unit_interval_scale(cv::InputArray read_mask = cv::noArray()) const;

        /**
         * @brief Resolves the endpoints of a range with negative values or a cv::Range::all().
         *
         * This function maps cv::Range::all() to cv::Range(0, levels()).
         * If `levels.start` (`levels.end`) is negative, it is mapped to the
         * `levels() + levels.start` (`levels() + levels.end`).
         *
         * @param[in] levels The range of levels.
         */
        cv::Range resolve_level_range(const cv::Range& levels) const
        {
            if (levels == cv::Range::all())
                return cv::Range(0, this->levels());

            return cv::Range(resolve_level(levels.start), resolve_level(levels.end));
        }
        ///@}

        bool has_same_metadata(const Coeffs& other) const
        {
            return _p->levels == other._p->levels
                && _p->image_size == other._p->image_size
                && _p->diagonal_subband_rects == other._p->diagonal_subband_rects
                && _p->wavelet == other._p->wavelet
                && _p->border_type == other._p->border_type;
        }
        /**
         * @private
         */
        friend std::vector<Coeffs> split(const Coeffs& coeffs);
        /**
         * @private
         */
        friend Coeffs merge(const std::vector<Coeffs>& coeffs);
        /**
         * @private
         */
        friend std::ostream& operator<<(std::ostream& stream, const Coeffs& wavelet);

    protected:
        //  Helpers
        /**
         * @brief Copies the source to the destination.
         *
         * @param[in] source A matrix or a scalar.
         * @param[in] destination The matrix that is written to.
         * @param[in] mask Indicates which destination elements are set.
         */
        template <typename Matrix>
        requires std::same_as<std::remove_cvref_t<Matrix>, cv::Mat>
        void convert_and_copy(
            cv::InputArray source,
            Matrix&& destination,
            cv::InputArray mask = cv::noArray()
        )
        {
            assert(destination.type() == type());

            if (is_scalar(source)) {
                destination.setTo(source, mask);
            } else if (source.type() != destination.type()) {
                cv::Mat converted;
                source.getMat().convertTo(converted, type());
                converted.copyTo(destination, mask);
            } else {
                source.copyTo(destination, mask);
            }
        }

        /**
         * @brief Maps a negative level index to the corresponding positive level index.
         *
         * @param[in] level
         */
        int resolve_level(int level) const { return (level >= 0) ? level : level + levels(); }

        /**
         * @brief Returns true if the array represents a scalar.
         *
         * In this context, a scalar is defined to be an object that can be used
         * to set a single coefficient (i.e. an element of the coefficient
         * matrix).
         *
         * A scalar is one of:
         *  - @copydetails common_scalar_definition_list
         *
         * In each case, `array.isVector()` is true.
         *
         * @param[in] array
         */
        bool is_scalar(cv::InputArray array) const
        {
            //  This is adapted from checkScalar in OpenCV source code.
            if (array.dims() > 2 || !array.isContinuous())
                return false;

            int channels = this->channels();

            return array.isVector()
                && (array.total() == channels || array.total() == 1)
                || (array.size() == cv::Size(1, 4) && array.type() == CV_64F && channels <= 4);
        }

    private:
        //  Argument Checkers - these can be disabled by building with cmake
        //  option CVWT_ENABLE_DWT2D_COEFFS_EXCEPTIONS = OFF
        void throw_if_bad_mask_for_normalize(cv::InputArray mask, const std::string mask_name) const CVWT_DWT2D_COEFFS_NOEXCEPT;
        void throw_if_wrong_size_for_assignment(cv::InputArray matrix) const CVWT_DWT2D_COEFFS_NOEXCEPT;
        void throw_if_wrong_size_for_set_level(cv::InputArray matrix, int level) const CVWT_DWT2D_COEFFS_NOEXCEPT;
        void throw_if_wrong_size_for_set_detail(cv::InputArray matrix, int level, int subband) const CVWT_DWT2D_COEFFS_NOEXCEPT;
        void throw_if_wrong_size_for_set_all_detail_levels(cv::InputArray matrix) const CVWT_DWT2D_COEFFS_NOEXCEPT;
        void throw_if_wrong_size_for_set_approx(cv::InputArray matrix) const CVWT_DWT2D_COEFFS_NOEXCEPT;
        void throw_if_level_out_of_range(int level) const CVWT_DWT2D_COEFFS_NOEXCEPT;
        void throw_if_levels_out_of_range(int lower_level, int upper_level) const CVWT_DWT2D_COEFFS_NOEXCEPT;
        void throw_if_this_is_empty() const CVWT_DWT2D_COEFFS_NOEXCEPT;
        void throw_if_invalid_subband(int subband) const CVWT_DWT2D_COEFFS_NOEXCEPT;

    private:
        std::shared_ptr<internal::Dwt2dCoeffsImpl> _p;
    };

public:
    /**
     * @brief Construct a new DWT2D object.
     *
     * @param[in] wavelet The wavelet.
     * @param[in] border_type The border exptrapolation method.
     */
    DWT2D(const Wavelet& wavelet, cv::BorderTypes border_type=cv::BORDER_DEFAULT);
    DWT2D() = delete;
    /**
     * @brief Copy Constructor.
     */
    DWT2D(const DWT2D& other) = default;
    /**
     * @brief Move Constructor.
     */
    DWT2D(DWT2D&& other) = default;

    /**
     * @brief Alias of decompose().
     */
    auto operator()(auto&&... args) const { return decompose(std::forward<decltype(args)>(args)...); }

    /**
     * @brief Performs a multiscale discrete wavelet transformation.
     *
     * @param[in] image The image to be transformed.
     * @param[in] levels
     */
    Coeffs decompose(cv::InputArray image, int levels) const
    {
        DWT2D::Coeffs coeffs;
        decompose(image, coeffs, levels);
        return coeffs;
    }

    /**
     * @overload
     *
     * @param[in]  image The image to be transformed.
     * @param[out] coeffs The result of the discrete wavelet transformation.
     * @param[in]  levels The number of levels.
     */
    void decompose(cv::InputArray image, Coeffs& coeffs, int levels) const;

    /**
     * @overload
     *
     * @param[in] image The image to be transformed.
     */
    Coeffs decompose(cv::InputArray image) const
    {
        Coeffs coeffs;
        decompose(image, coeffs);
        return coeffs;
    }

    /**
     * @overload
     *
     * @param[in]  image The image to be transformed.
     * @param[out] coeffs The result of the discrete wavelet transformation.
     */
    void decompose(cv::InputArray image, Coeffs& coeffs) const
    {
        decompose(image, coeffs, max_levels_without_border_effects(image));
    }

    /**
     * @brief Reconstructs an image from DWT coefficients.
     *
     * This performs an inverse multilevel discrete wavelet transformation.
     *
     * @param[in] coeffs
     */
    cv::Mat reconstruct(const Coeffs& coeffs) const
    {
        cv::Mat image;
        reconstruct(coeffs, image);
        return image;
    }

    /**
     * @overload
     *
     * @param[in]  coeffs The discrete wavelet transform coefficients.
     * @param[out] image The reconstructed image.
     */
    void reconstruct(const Coeffs& coeffs, cv::OutputArray image) const;

    /**
     * @brief Creates a DWT2D::Coeffs object.
     *
     * @param[in] coeffs_matrix The initial discrete wavelet transform coefficients.
     * @param[in] image_size The size of the reconstructed image.
     * @param[in] levels The number of levels.
     */
    Coeffs create_coeffs(
        cv::InputArray coeffs_matrix,
        const cv::Size& image_size,
        int levels
    ) const;

    /**
     * @brief Creates an empty DWT2D::Coeffs object.
     *
     * @param[in] image_size The size of the reconstructed image.
     * @param[in] levels The number of levels.
     */
    Coeffs create_empty_coeffs(
        const cv::Size& image_size,
        int levels
    ) const;

    /**
     * @brief Creates a zero initialized DWT2D::Coeffs object.
     *
     * @param[in] image_size The size of the reconstructed image.
     * @param[in] type The type of the reconstructed image.
     * @param[in] levels The number of levels.
     */
    Coeffs create_coeffs(const cv::Size& image_size, int type, int levels) const;

    /**
     * @brief Creates a zero initialized DWT2D::Coeffs object.
     *
     * This is a overloaded member function, provided for convenience.
     * It is equivalent to:
     * @code{cpp}
     * this->create_coeffs(image.size(), image.type(), levels);
     * @endcode
     *
     * @param[in] image
     * @param[in] levels
     */
    Coeffs create_coeffs(cv::InputArray image, int levels) const
    {
        return create_coeffs(image.size(), image.type(), levels);
    }

    /**
     * @brief Creates a zero initialized DWT2D::Coeffs object.
     *
     * This is a overloaded member function, provided for convenience.
     * It is equivalent to:
     * @code{cpp}
     * this->create_coeffs(cv::Size(image_cols, image_rows), type, levels);
     * @endcode
     *
     * @param[in] image_rows
     * @param[in] image_cols
     * @param[in] type
     * @param[in] levels
     */
    Coeffs create_coeffs(int image_rows, int image_cols, int type, int levels) const
    {
        return create_coeffs(cv::Size(image_cols, image_rows), type, levels);
    }

    /**
     * @brief Returns the size of the DWT2D::Coeffs required to perfectly represent the given image size.
     *
     * Decomposing an image typically produces a coefficients matrix that is
     * larger than the image itself because the filter bank must extrapolate the
     * image along the border.
     *
     * The size of the multiscale decomposition coefficients is a function
     * of the input size, the number of levels, and the Wavelet::filter_length().
     *
     * @see FilterBank::subband_size()
     *
     * @param[in] image_size
     * @param[in] levels
     */
    cv::Size coeffs_size_for_image(const cv::Size& image_size, int levels) const;

    /**
     * @brief Returns the size of the DWT2D::Coeffs required to perfectly represent the given image.
     *
     * This is a overloaded member function, provided for convenience.
     * It is equivalent to:
     * @code{cpp}
     * this->coeffs_size_for_image(image.size(), levels);
     * @endcode
     *
     * @see coeffs_size_for_image(const cv::Size& image_size, int levels) const
     *
     * @param[in] image
     * @param[in] levels
     */
    cv::Size coeffs_size_for_image(cv::InputArray image, int levels) const
    {
        return coeffs_size_for_image(image.size(), levels);
    }

    /**
     * @brief Returns the size of the DWT2D::Coeffs required to perfectly represent the given image size.
     *
     * This is a overloaded member function, provided for convenience.
     * It is equivalent to:
     * @code{cpp}
     * this->coeffs_size_for_image(cv::Size(image_cols, image_rows), levels);
     * @endcode
     *
     * @see coeffs_size_for_image(const cv::Size& image_size, int levels) const
     *
     * @param[in] image_rows
     * @param[in] image_cols
     * @param[in] levels
     */
    cv::Size coeffs_size_for_image(int image_rows, int image_cols, int levels) const
    {
        return coeffs_size_for_image(cv::Size(image_cols, image_rows), levels);
    }

    /**
     * @brief Returns the maximum number of decomposition levels possible while maintaining perfect reconstruction.
     *
     * @param[in] image_size
     */
    int max_levels_without_border_effects(const cv::Size& image_size) const
    {
        return max_levels_without_border_effects(image_size.height, image_size.width);
    }

    /**
     * @brief Returns the maximum number of decomposition levels possible while maintaining perfect reconstruction.
     *
     * @param[in] image_rows
     * @param[in] image_cols
     */
    int max_levels_without_border_effects(int image_rows, int image_cols) const;

    /**
     * @brief Returns the maximum number of decomposition levels possible while maintaining perfect reconstruction.
     *
     * @param[in] image
     */
    int max_levels_without_border_effects(cv::InputArray image) const
    {
        return max_levels_without_border_effects(image.size());
    }
private:
    //  Argument Checkers - these can be disabled by building with cmake
    //  option CVWT_ENABLE_DWT2D_EXCEPTIONS = OFF
    void throw_if_levels_out_of_range(int levels) const CVWT_DWT2D_NOEXCEPT;
    void throw_if_inconsistent_coeffs_and_image_sizes(
        cv::InputArray coeffs,
        const cv::Size& image_size,
        int levels
    ) const CVWT_DWT2D_NOEXCEPT;

    //  Log warnings - these can be disabled by defining CVWT_DISABLE_DWT_WARNINGS_ENABLED
    void warn_if_border_effects_will_occur(int levels, const cv::Size& image_size) const noexcept;
    void warn_if_border_effects_will_occur(int levels, cv::InputArray image) const noexcept;
    void warn_if_border_effects_will_occur(const Coeffs& coeffs) const noexcept;

    std::vector<cv::Size> calc_subband_sizes(const cv::Size& image_size, int levels) const;
public:
    Wavelet wavelet;
    cv::BorderTypes border_type;
};

/**
 * @brief Splits DWT2D::Coeffs into separate channels.
 *
 * @param[in] coeffs
 */
std::vector<DWT2D::Coeffs> split(const DWT2D::Coeffs& coeffs);

/**
 * @brief Merges separate DWT2D::Coeffs channels.
 *
 * @param[in] coeffs
 */
DWT2D::Coeffs merge(const std::vector<DWT2D::Coeffs>& coeffs);

/**
 * @brief Writes a string representation of a DWT2D::Coeffs to an output stream.
 */
std::ostream& operator<<(std::ostream& stream, const DWT2D::Coeffs& coeffs);



//  ----------------------------------------------------------------------------
//  Functional Interface
//  ----------------------------------------------------------------------------
/** @{ DWT Functional API */
/**
 * @brief Performs a multiscale discrete wavelet transform.
 *
 * This convenience wrapper around a DWT2D object.
 * It is equivalent to:
 * @code{cpp}
 * DWT2D dwt(wavelet, border_type);
 * return dwt.decompose(image, levels);
 * @endcode
 *
 * @see
 *   - DWT2D
 *   - idwt2d()
 *
 * @param[in] image
 * @param[in] wavelet
 * @param[in] levels
 * @param[in] border_type
 */
DWT2D::Coeffs dwt2d(
    cv::InputArray image,
    const Wavelet& wavelet,
    int levels,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

/**
 * @overload
 *
 * This convenience wrapper around a DWT2D object.
 * It is equivalent to:
 * @code{cpp}
 * DWT2D dwt(wavelet, border_type);
 * dwt.decompose(image, output, levels);
 * @endcode
 *
 * @see
 *   - DWT2D
 *   - idwt2d()
 *
 * @param[in] image
 * @param[out] coeffs
 * @param[in] wavelet
 * @param[in] levels
 * @param[in] border_type
 */
void dwt2d(
    cv::InputArray image,
    DWT2D::Coeffs& coeffs,
    const Wavelet& wavelet,
    int levels,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

/**
 * @overload
 *
 * This is a overloaded function, provided for convenience.
 * It is equivalent to:
 * @code{cpp}
 * dwt2d(image, Wavelet::create(wavelet), levels, border_type);
 * @endcode
 *
 * @see
 *   - DWT2D
 *   - idwt2d()
 *
 * @param[in] image
 * @param[in] wavelet
 * @param[in] levels
 * @param[in] border_type
 */
DWT2D::Coeffs dwt2d(
    cv::InputArray image,
    const std::string& wavelet,
    int levels,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

/**
 * @overload
 *
 * This is a overloaded function, provided for convenience.
 * It is equivalent to:
 * @code{cpp}
 * dwt2d(image, output, Wavelet::create(wavelet), levels, border_type);
 * @endcode
 *
 * @see
 *   - DWT2D
 *   - idwt2d()
 *
 * @param[in] image
 * @param[out] coeffs
 * @param[in] wavelet
 * @param[in] levels
 * @param[in] border_type
 */
void dwt2d(
    cv::InputArray image,
    DWT2D::Coeffs& coeffs,
    const std::string& wavelet,
    int levels,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

/**
 * @overload
 *
 * This convenience wrapper around a DWT2D object.
 * It is equivalent to:
 * @code{cpp}
 * DWT2D dwt(wavelet, border_type);
 * return dwt.decompose(image);
 * @endcode
 *
 * @see idwt2d()
 *
 * @param[in] image
 * @param[in] wavelet
 * @param[in] border_type
 */
DWT2D::Coeffs dwt2d(
    cv::InputArray image,
    const Wavelet& wavelet,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

/**
 * @overload
 *
 * This convenience wrapper around a DWT2D object.
 * It is equivalent to:
 * @code{cpp}
 * DWT2D dwt(wavelet, border_type);
 * dwt.decompose(image, output);
 * @endcode
 *
 * @see
 *   - DWT2D
 *   - idwt2d()
 *
 * @param[in] image
 * @param[out] coeffs
 * @param[in] wavelet
 * @param[in] border_type
 */
void dwt2d(
    cv::InputArray image,
    DWT2D::Coeffs& coeffs,
    const Wavelet& wavelet,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

/**
 * @overload
 *
 * This is a overloaded member function, provided for convenience.
 * It is equivalent to:
 * @code{cpp}
 * dwt2d(image, Wavelet::create(wavelet), border_type);
 * @endcode
 *
 * @see
 *   - DWT2D
 *   - idwt2d()
 *
 * @param[in] image
 * @param[in] wavelet
 * @param[in] border_type
 */
DWT2D::Coeffs dwt2d(
    cv::InputArray image,
    const std::string& wavelet,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

/**
 * @overload
 *
 * This is a overloaded function, provided for convenience.
 * It is equivalent to:
 * @code{cpp}
 * dwt2d(image, output, Wavelet::create(wavelet), border_type);
 * @endcode
 *
 * @see
 *   - DWT2D
 *   - idwt2d()
 *
 * @param[in] image
 * @param[out] coeffs
 * @param[in] wavelet
 * @param[in] border_type
 */
void dwt2d(
    cv::InputArray image,
    DWT2D::Coeffs& coeffs,
    const std::string& wavelet,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

/**
 * @brief Reconstructs an image from DWT coefficients.
 *
 * This convenience wrapper around a DWT2D object.
 * It is equivalent to:
 * @code{cpp}
 * DWT2D dwt(wavelet, border_type);
 * return dwt.reconstruct(coeffs);
 * @endcode
 *
 * @see
 *   - DWT2D
 *   - dwt2d()
 *
 * @param[in] coeffs
 * @param[in] wavelet
 * @param[in] border_type
 */
cv::Mat idwt2d(
    const DWT2D::Coeffs& coeffs,
    const Wavelet& wavelet,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

/**
 * @overload
 *
 * This convenience wrapper around a DWT2D object.
 * It is equivalent to:
 * @code{cpp}
 * DWT2D dwt(wavelet, border_type);
 * dwt.reconstruct(coeffs, output);
 * @endcode
 *
 * @see
 *   - DWT2D
 *   - idwt2d()
 *
 * @param[in] coeffs
 * @param[out] image
 * @param[in] wavelet
 * @param[in] border_type
 */
void idwt2d(
    const DWT2D::Coeffs& coeffs,
    cv::OutputArray image,
    const Wavelet& wavelet,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

/**
 * @brief Reconstructs an image from DWT coefficients.
 *
 * This is a overloaded function, provided for convenience.
 * It is equivalent to:
 * @code{cpp}
 * idwt2d(coeffs, Wavelet::create(wavelet), border_type);
 * @endcode
 *
 * @see
 *   - DWT2D
 *   - dwt2d()
 *
 * @param[in] coeffs
 * @param[in] wavelet
 * @param[in] border_type
 */
cv::Mat idwt2d(
    const DWT2D::Coeffs& coeffs,
    const std::string& wavelet,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

/**
 * @overload
 *
 * This is a overloaded function, provided for convenience.
 * It is equivalent to:
 * @code{cpp}
 * idwt2d(coeffs, output, Wavelet::create(wavelet), border_type);
 * @endcode
 *
 * @see
 *   - DWT2D
 *   - dwt2d()
 *
 * @param[in] coeffs
 * @param[out] image
 * @param[in] wavelet
 * @param[in] border_type
 */
void idwt2d(
    const DWT2D::Coeffs& coeffs,
    cv::OutputArray image,
    const std::string& wavelet,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);
/** @} DWT Functional API*/
/** @} dwt2d*/

//  ----------------------------------------------------------------------------
//  Expressions
class CoeffsExpr : public cv::MatExpr
{
public:
    CoeffsExpr() :
        cv::MatExpr(),
        coeffs()
    {}

    explicit CoeffsExpr(const DWT2D::Coeffs& coeffs) :
        cv::MatExpr(static_cast<const cv::Mat&>(coeffs)),
        coeffs(coeffs)
    {}

    CoeffsExpr(
        const DWT2D::Coeffs& coeffs,
        const cv::MatExpr& expression
    ) :
        cv::MatExpr(
            expression.op,
            expression.flags,
            expression.a,
            expression.b,
            expression.c,
            expression.alpha,
            expression.beta,
            expression.s
        ),
        coeffs(coeffs)
    {}

    CoeffsExpr(
        const DWT2D::Coeffs& coeffs_a,
        const DWT2D::Coeffs& coeffs_b,
        const cv::MatExpr& expression
    ) :
        cv::MatExpr(
            expression.op,
            expression.flags,
            expression.a,
            expression.b,
            expression.c,
            expression.alpha,
            expression.beta,
            expression.s
        ),
        coeffs(coeffs_a)
    {
        throw_if_incompatible(coeffs_a, coeffs_b);
    }

    operator DWT2D::Coeffs() const
    {
        return coeffs.clone_and_assign(*this);
    }

    CoeffsExpr mul(const DWT2D::Coeffs& other, double scale = 1.0) const
    {
        return CoeffsExpr(coeffs, other, MatExpr::mul(other));
    }

    CoeffsExpr mul(const CoeffsExpr& other, double scale = 1.0) const
    {
        return CoeffsExpr(coeffs, other.coeffs, MatExpr::mul(other, scale));
    }

    CoeffsExpr mul(const cv::Mat& other, double scale = 1.0) const
    {
        return CoeffsExpr(coeffs, MatExpr::mul(other, scale));
    }

    template <typename T, int m, int n>
    CoeffsExpr mul(const cv::Matx<T, m, n>& other, double scale = 1.0) const
    {
        return CoeffsExpr(coeffs, MatExpr::mul(cv::Mat(other), scale));
    }

    CoeffsExpr mul(const cv::MatExpr& other, double scale = 1.0) const
    {
        return CoeffsExpr(coeffs, MatExpr::mul(other, scale));
    }

    void swap(CoeffsExpr& other)
    {
        MatExpr::swap(other);
        auto tmp = coeffs;
        coeffs = other.coeffs;
        other.coeffs = tmp;
    }

    static void throw_if_incompatible(
        const DWT2D::Coeffs& coeffs_a,
        const DWT2D::Coeffs& coeffs_b
    ) CVWT_DWT2D_COEFFS_NOEXCEPT;

public:
    DWT2D::Coeffs coeffs;
};

//  ============================================================================
//  Arithmetic
//  ============================================================================
//  ----------------------------------------------------------------------------
//  negation
CoeffsExpr operator-(const DWT2D::Coeffs& coeffs);
CoeffsExpr operator-(const CoeffsExpr& expression);

//  ----------------------------------------------------------------------------
//  addition
CoeffsExpr operator+(const CoeffsExpr& lhs, const CoeffsExpr& rhs);

CoeffsExpr operator+(const DWT2D::Coeffs& lhs, const DWT2D::Coeffs& rhs);

CoeffsExpr operator+(const CoeffsExpr& lhs, const DWT2D::Coeffs& rhs);
inline CoeffsExpr operator+(const DWT2D::Coeffs& lhs, const CoeffsExpr& rhs)
{
    return rhs + lhs;
}

CoeffsExpr operator+(const CoeffsExpr& lhs, const cv::MatExpr& rhs);
inline CoeffsExpr operator+(const cv::MatExpr& lhs, const CoeffsExpr& rhs)
{
    return rhs + lhs;
}

CoeffsExpr operator+(const CoeffsExpr& lhs, const cv::Mat& rhs);
inline CoeffsExpr operator+(const cv::Mat& lhs, const CoeffsExpr& rhs)
{
    return rhs + lhs;
}

template <typename T, int m, int n>
CoeffsExpr operator+(const CoeffsExpr& lhs, const cv::Matx<T, m, n>& rhs)
{
    return CoeffsExpr(lhs.coeffs, static_cast<const cv::MatExpr&>(lhs) + rhs);
}
template <typename T, int m, int n>
CoeffsExpr operator+(const cv::Matx<T, m, n>& lhs, const CoeffsExpr& rhs)
{
    return rhs + lhs;
}

CoeffsExpr operator+(const CoeffsExpr& lhs, const cv::Scalar& scalar);
inline CoeffsExpr operator+(const cv::Scalar& lhs, const CoeffsExpr& rhs)
{
    return rhs + lhs;
}

CoeffsExpr operator+(const DWT2D::Coeffs& lhs, const cv::MatExpr& rhs);
inline CoeffsExpr operator+(const cv::MatExpr& lhs, const DWT2D::Coeffs& rhs)
{
    return rhs + lhs;
}

CoeffsExpr operator+(const DWT2D::Coeffs& lhs, const cv::Mat& rhs);
inline CoeffsExpr operator+(const cv::Mat& lhs, const DWT2D::Coeffs& rhs)
{
    return rhs + lhs;
}

template <typename T, int m, int n>
CoeffsExpr operator+(const DWT2D::Coeffs& lhs, const cv::Matx<T, m, n>& rhs)
{
    return CoeffsExpr(lhs, static_cast<const cv::Mat&>(lhs) + rhs);
}
template <typename T, int m, int n>
CoeffsExpr operator+(const cv::Matx<T, m, n>& lhs, const DWT2D::Coeffs& rhs)
{
    return rhs + lhs;
}

CoeffsExpr operator+(const DWT2D::Coeffs& lhs, const cv::Scalar& scalar);
inline CoeffsExpr operator+(const cv::Scalar& lhs, const DWT2D::Coeffs& rhs)
{
    return rhs + lhs;
}

//  ----------------------------------------------------------------------------
//  subtraction
CoeffsExpr operator-(const CoeffsExpr& lhs, const CoeffsExpr& rhs);

CoeffsExpr operator-(const DWT2D::Coeffs& lhs, const DWT2D::Coeffs& rhs);

CoeffsExpr operator-(const CoeffsExpr& lhs, const DWT2D::Coeffs& rhs);
CoeffsExpr operator-(const DWT2D::Coeffs& lhs, const CoeffsExpr& rhs);

CoeffsExpr operator-(const CoeffsExpr& lhs, const cv::MatExpr& rhs);
CoeffsExpr operator-(const cv::MatExpr& rhs, const CoeffsExpr& lhs);

CoeffsExpr operator-(const CoeffsExpr& lhs, const cv::Mat& rhs);
CoeffsExpr operator-(const cv::Mat& lhs, const CoeffsExpr& rhs);

template <typename T, int m, int n>
CoeffsExpr operator-(const CoeffsExpr& lhs, const cv::Matx<T, m, n>& rhs)
{
    return CoeffsExpr(lhs.coeffs, static_cast<const cv::MatExpr&>(lhs) - rhs);
}
template <typename T, int m, int n>
CoeffsExpr operator-(const cv::Matx<T, m, n>& lhs, const CoeffsExpr& rhs)
{
    return CoeffsExpr(rhs.coeffs, lhs - static_cast<const cv::MatExpr&>(rhs));
}

CoeffsExpr operator-(const CoeffsExpr& lhs, const cv::Scalar& rhs);
CoeffsExpr operator-(const cv::Scalar& lhs, const CoeffsExpr& rhs);

CoeffsExpr operator-(const DWT2D::Coeffs& lhs, const cv::MatExpr& rhs);
CoeffsExpr operator-(const cv::MatExpr& lhs, const DWT2D::Coeffs& rhs);

CoeffsExpr operator-(const DWT2D::Coeffs& lhs, const cv::Mat& rhs);
CoeffsExpr operator-(const cv::Mat& lhs, const DWT2D::Coeffs& rhs);

template <typename T, int m, int n>
CoeffsExpr operator-(const DWT2D::Coeffs& lhs, const cv::Matx<T, m, n>& rhs)
{
    return CoeffsExpr(lhs, static_cast<const cv::Mat&>(lhs) - rhs);
}
template <typename T, int m, int n>
CoeffsExpr operator-(const cv::Matx<T, m, n>& lhs, const DWT2D::Coeffs& rhs)
{
    return CoeffsExpr(rhs, lhs - static_cast<const cv::Mat&>(rhs));
}

CoeffsExpr operator-(const DWT2D::Coeffs& lhs, const cv::Scalar& rhs);
CoeffsExpr operator-(const cv::Scalar& lhs, const DWT2D::Coeffs& rhs);

//  ----------------------------------------------------------------------------
//  multiplication
CoeffsExpr operator*(const CoeffsExpr& lhs, const CoeffsExpr& rhs);

CoeffsExpr operator*(const DWT2D::Coeffs& lhs, const DWT2D::Coeffs& rhs);

CoeffsExpr operator*(const CoeffsExpr& lhs, const DWT2D::Coeffs& rhs);
inline CoeffsExpr operator*(const DWT2D::Coeffs& lhs, const CoeffsExpr& rhs)
{
    return rhs * lhs;
}

CoeffsExpr operator*(const CoeffsExpr& lhs, const cv::MatExpr& rhs);
inline CoeffsExpr operator*(const cv::MatExpr& lhs, const CoeffsExpr& rhs)
{
    return rhs * lhs;
}

CoeffsExpr operator*(const CoeffsExpr& lhs, const cv::Mat& rhs);
inline CoeffsExpr operator*(const cv::Mat& lhs, const CoeffsExpr& rhs)
{
    return rhs * lhs;
}

template <typename T, int m, int n>
CoeffsExpr operator*(const CoeffsExpr& lhs, const cv::Matx<T, m, n>& rhs)
{
    return lhs.mul(rhs);
}
template <typename T, int m, int n>
CoeffsExpr operator*(const cv::Matx<T, m, n>& lhs, const CoeffsExpr& rhs)
{
    return rhs * lhs;
}

CoeffsExpr operator*(const CoeffsExpr& lhs, double rhs);
inline CoeffsExpr operator*(double lhs, const CoeffsExpr& rhs)
{
    return rhs * lhs;
}

CoeffsExpr operator*(const DWT2D::Coeffs& lhs, const cv::MatExpr& rhs);
inline CoeffsExpr operator*(const cv::MatExpr& lhs, const DWT2D::Coeffs& rhs)
{
    return rhs * lhs;
}

CoeffsExpr operator*(const DWT2D::Coeffs& lhs, const cv::Mat& rhs);
inline CoeffsExpr operator*(const cv::Mat& lhs, const DWT2D::Coeffs& rhs)
{
    return rhs * lhs;
}

template <typename T, int m, int n>
CoeffsExpr operator*(const DWT2D::Coeffs& lhs, const cv::Matx<T, m, n>& rhs)
{
    return lhs.mul(rhs);
}
template <typename T, int m, int n>
CoeffsExpr operator*(const cv::Matx<T, m, n>& lhs, const DWT2D::Coeffs& rhs)
{
    return rhs * lhs;
}

CoeffsExpr operator*(const DWT2D::Coeffs& lhs, double rhs);
inline CoeffsExpr operator*(double lhs, const DWT2D::Coeffs& rhs)
{
    return rhs * lhs;
}

//  ----------------------------------------------------------------------------
//  division
CoeffsExpr operator/(const CoeffsExpr& lhs, const CoeffsExpr& rhs);

CoeffsExpr operator/(const DWT2D::Coeffs& lhs, const DWT2D::Coeffs& rhs);

CoeffsExpr operator/(const CoeffsExpr& lhs, const DWT2D::Coeffs& rhs);
CoeffsExpr operator/(const DWT2D::Coeffs& lhs, const CoeffsExpr& rhs);

CoeffsExpr operator/(const CoeffsExpr& lhs, const cv::MatExpr& rhs);
CoeffsExpr operator/(const cv::MatExpr& lhs, const CoeffsExpr& rhs);

CoeffsExpr operator/(const CoeffsExpr& lhs, const cv::Mat& rhs);
CoeffsExpr operator/(const cv::Mat& lhs, const CoeffsExpr& rhs);

template <typename T, int m, int n>
CoeffsExpr operator/(const CoeffsExpr& lhs, const cv::Matx<T, m, n>& rhs)
{
    return CoeffsExpr(lhs.coeffs, static_cast<const cv::MatExpr&>(lhs) / rhs);
}
template <typename T, int m, int n>
CoeffsExpr operator/(const cv::Matx<T, m, n>& lhs, const CoeffsExpr& rhs)
{
    return CoeffsExpr(rhs.coeffs, lhs / static_cast<const cv::MatExpr&>(rhs));
}

CoeffsExpr operator/(const CoeffsExpr& lhs, double rhs);
CoeffsExpr operator/(double lhs, const CoeffsExpr& rhs);

CoeffsExpr operator/(const DWT2D::Coeffs& lhs, const cv::MatExpr& rhs);
CoeffsExpr operator/(const cv::MatExpr& lhs, const DWT2D::Coeffs& rhs);

CoeffsExpr operator/(const DWT2D::Coeffs& lhs, const cv::Mat& rhs);
CoeffsExpr operator/(const cv::Mat& lhs, const DWT2D::Coeffs& rhs);

template <typename T, int m, int n>
CoeffsExpr operator/(const DWT2D::Coeffs& lhs, const cv::Matx<T, m, n>& rhs)
{
    return CoeffsExpr(lhs, static_cast<const cv::Mat&>(lhs) / rhs);
}
template <typename T, int m, int n>
CoeffsExpr operator/(const cv::Matx<T, m, n>& lhs, const DWT2D::Coeffs& rhs)
{
    return CoeffsExpr(rhs, lhs / static_cast<const cv::Mat&>(rhs));
}

CoeffsExpr operator/(const DWT2D::Coeffs& lhs, double rhs);
CoeffsExpr operator/(double lhs, const DWT2D::Coeffs& rhs);


//  ============================================================================
//  Comparison
//  ============================================================================
namespace internal
{
class CompareOp : public cv::MatOp
{
public:
    enum Flags {
        A_IS_SCALAR = (1 << 8),
        B_IS_SCALAR = (2 << 8),
    };

public:
    void assign(const cv::MatExpr& expression, cv::Mat& m, int _type = -1) const override;

    static void make_expression(
        cv::MatExpr& expression,
        cv::CmpTypes cmp_type,
        const cv::Mat& a,
        const cv::Mat& b,
        int flags = 0
    );
    static void make_expression(
        cv::MatExpr& expression,
        cv::CmpTypes cmp_type,
        const cv::Mat& a,
        double b
    );
    static void make_expression(
        cv::MatExpr& expression,
        cv::CmpTypes cmp_type,
        double a,
        const cv::Mat& b
    );
};
}   // namespace internal

//  ----------------------------------------------------------------------------
//  equal
cv::MatExpr operator==(const CoeffsExpr& lhs, const CoeffsExpr& rhs);

cv::MatExpr operator==(const DWT2D::Coeffs& lhs, const DWT2D::Coeffs& rhs);

cv::MatExpr operator==(const CoeffsExpr& lhs, const DWT2D::Coeffs& rhs);
inline cv::MatExpr operator==(const DWT2D::Coeffs& lhs, const CoeffsExpr& rhs)
{
    return rhs == lhs;
}

cv::MatExpr operator==(const CoeffsExpr& lhs, const cv::MatExpr& rhs);
inline cv::MatExpr operator==(const cv::MatExpr& lhs, const CoeffsExpr& rhs)
{
    return rhs == lhs;
}

cv::MatExpr operator==(const CoeffsExpr& lhs, const cv::Mat& rhs);
inline cv::MatExpr operator==(const cv::Mat& lhs, const CoeffsExpr& rhs)
{
    return rhs == lhs;
}

template <typename T, int m, int n>
cv::MatExpr operator==(const CoeffsExpr& lhs, const cv::Matx<T, m, n>& rhs)
{
    cv::MatExpr compare_expression;
    internal::CompareOp::make_expression(
        compare_expression,
        cv::CMP_EQ,
        lhs.coeffs,
        cv::Mat(rhs),
        is_scalar_for_array(rhs, lhs.coeffs) ? internal::CompareOp::B_IS_SCALAR : 0
    );
    return compare_expression;
}
template <typename T, int m, int n>
cv::MatExpr operator==(const cv::Matx<T, m, n>& lhs, const CoeffsExpr& rhs)
{
    return rhs == lhs;
}

cv::MatExpr operator==(const CoeffsExpr& lhs, double rhs);
inline cv::MatExpr operator==(double lhs, const CoeffsExpr& rhs)
{
    return rhs == lhs;
}

cv::MatExpr operator==(const DWT2D::Coeffs& lhs, const cv::MatExpr& rhs);
inline cv::MatExpr operator==(const cv::MatExpr& lhs, const DWT2D::Coeffs& rhs)
{
    return rhs == lhs;
}

cv::MatExpr operator==(const DWT2D::Coeffs& lhs, const cv::Mat& rhs);
inline cv::MatExpr operator==(const cv::Mat& lhs, const DWT2D::Coeffs& rhs)
{
    return rhs == lhs;
}

template <typename T, int m, int n>
cv::MatExpr operator==(const DWT2D::Coeffs& lhs, const cv::Matx<T, m, n>& rhs)
{
    cv::MatExpr compare_expression;
    internal::CompareOp::make_expression(
        compare_expression,
        cv::CMP_EQ,
        lhs,
        cv::Mat(rhs),
        is_scalar_for_array(rhs, lhs) ? internal::CompareOp::B_IS_SCALAR : 0
    );
    return compare_expression;
}
template <typename T, int m, int n>
cv::MatExpr operator==(const cv::Matx<T, m, n>& lhs, const DWT2D::Coeffs& rhs)
{
    return rhs == lhs;
}

cv::MatExpr operator==(const DWT2D::Coeffs& lhs, double rhs);
inline cv::MatExpr operator==(double lhs, const DWT2D::Coeffs& rhs)
{
    return rhs == lhs;
}

//  ----------------------------------------------------------------------------
//  not equal
cv::MatExpr operator!=(const CoeffsExpr& lhs, const CoeffsExpr& rhs);

cv::MatExpr operator!=(const DWT2D::Coeffs& lhs, const DWT2D::Coeffs& rhs);

cv::MatExpr operator!=(const CoeffsExpr& lhs, const DWT2D::Coeffs& rhs);
inline cv::MatExpr operator!=(const DWT2D::Coeffs& lhs, const CoeffsExpr& rhs)
{
    return rhs != lhs;
}

cv::MatExpr operator!=(const CoeffsExpr& lhs, const cv::MatExpr& rhs);
inline cv::MatExpr operator!=(const cv::MatExpr& lhs, const CoeffsExpr& rhs)
{
    return rhs != lhs;
}

cv::MatExpr operator!=(const CoeffsExpr& lhs, const cv::Mat& rhs);
inline cv::MatExpr operator!=(const cv::Mat& lhs, const CoeffsExpr& rhs)
{
    return rhs != lhs;
}

template <typename T, int m, int n>
cv::MatExpr operator!=(const CoeffsExpr& lhs, const cv::Matx<T, m, n>& rhs)
{
    cv::MatExpr compare_expression;
    internal::CompareOp::make_expression(
        compare_expression,
        cv::CMP_NE,
        lhs.coeffs,
        cv::Mat(rhs),
        is_scalar_for_array(rhs, lhs.coeffs) ? internal::CompareOp::B_IS_SCALAR : 0
    );
    return compare_expression;
}
template <typename T, int m, int n>
cv::MatExpr operator!=(const cv::Matx<T, m, n>& lhs, const CoeffsExpr& rhs)
{
    return rhs != lhs;
}

cv::MatExpr operator!=(const CoeffsExpr& lhs, double rhs);
inline cv::MatExpr operator!=(double lhs, const CoeffsExpr& rhs)
{
    return rhs != lhs;
}

cv::MatExpr operator!=(const DWT2D::Coeffs& lhs, const cv::MatExpr& rhs);
inline cv::MatExpr operator!=(const cv::MatExpr& lhs, const DWT2D::Coeffs& rhs)
{
    return rhs != lhs;
}

cv::MatExpr operator!=(const DWT2D::Coeffs& lhs, const cv::Mat& rhs);
inline cv::MatExpr operator!=(const cv::Mat& lhs, const DWT2D::Coeffs& rhs)
{
    return rhs != lhs;
}

template <typename T, int m, int n>
cv::MatExpr operator!=(const DWT2D::Coeffs& lhs, const cv::Matx<T, m, n>& rhs)
{
    cv::MatExpr compare_expression;
    internal::CompareOp::make_expression(
        compare_expression,
        cv::CMP_NE,
        lhs,
        cv::Mat(rhs),
        is_scalar_for_array(rhs, lhs) ? internal::CompareOp::B_IS_SCALAR : 0
    );
    return compare_expression;
}
template <typename T, int m, int n>
cv::MatExpr operator!=(const cv::Matx<T, m, n>& lhs, const DWT2D::Coeffs& rhs)
{
    return rhs != lhs;
}

cv::MatExpr operator!=(const DWT2D::Coeffs& lhs, double rhs);
inline cv::MatExpr operator!=(double lhs, const DWT2D::Coeffs& rhs)
{
    return rhs != lhs;
}

//  ----------------------------------------------------------------------------
//  less than
cv::MatExpr operator<(const CoeffsExpr& lhs, const CoeffsExpr& rhs);

cv::MatExpr operator<(const DWT2D::Coeffs& lhs, const DWT2D::Coeffs& rhs);

cv::MatExpr operator<(const CoeffsExpr& lhs, const DWT2D::Coeffs& rhs);
cv::MatExpr operator<(const DWT2D::Coeffs& lhs, const CoeffsExpr& rhs);

cv::MatExpr operator<(const CoeffsExpr& lhs, const cv::MatExpr& rhs);
cv::MatExpr operator<(const cv::MatExpr& lhs, const CoeffsExpr& rhs);

cv::MatExpr operator<(const CoeffsExpr& lhs, const cv::Mat& rhs);
cv::MatExpr operator<(const cv::Mat& lhs, const CoeffsExpr& rhs);

template <typename T, int m, int n>
cv::MatExpr operator<(const CoeffsExpr& lhs, const cv::Matx<T, m, n>& rhs)
{
    cv::MatExpr compare_expression;
    internal::CompareOp::make_expression(
        compare_expression,
        cv::CMP_LT,
        lhs.coeffs,
        cv::Mat(rhs),
        is_scalar_for_array(rhs, lhs.coeffs) ? internal::CompareOp::B_IS_SCALAR : 0
    );
    return compare_expression;
}
template <typename T, int m, int n>
cv::MatExpr operator<(const cv::Matx<T, m, n>& lhs, const CoeffsExpr& rhs)
{
    return rhs > lhs;
}

cv::MatExpr operator<(const CoeffsExpr& lhs, double rhs);
cv::MatExpr operator<(double lhs, const CoeffsExpr& rhs);

cv::MatExpr operator<(const DWT2D::Coeffs& lhs, const cv::MatExpr& rhs);
cv::MatExpr operator<(const cv::MatExpr& lhs, const DWT2D::Coeffs& rhs);

cv::MatExpr operator<(const DWT2D::Coeffs& lhs, const cv::Mat& rhs);
cv::MatExpr operator<(const cv::Mat& lhs, const DWT2D::Coeffs& rhs);

template <typename T, int m, int n>
cv::MatExpr operator<(const DWT2D::Coeffs& lhs, const cv::Matx<T, m, n>& rhs)
{
    cv::MatExpr compare_expression;
    internal::CompareOp::make_expression(
        compare_expression,
        cv::CMP_LT,
        lhs,
        cv::Mat(rhs),
        is_scalar_for_array(rhs, lhs) ? internal::CompareOp::B_IS_SCALAR : 0
    );
    return compare_expression;
}
template <typename T, int m, int n>
cv::MatExpr operator<(const cv::Matx<T, m, n>& lhs, const DWT2D::Coeffs& rhs)
{
    return rhs > lhs;
}

cv::MatExpr operator<(const DWT2D::Coeffs& lhs, double rhs);
cv::MatExpr operator<(double lhs, const DWT2D::Coeffs& rhs);

//  ----------------------------------------------------------------------------
//  less than or equal
cv::MatExpr operator<=(const CoeffsExpr& lhs, const CoeffsExpr& rhs);

cv::MatExpr operator<=(const DWT2D::Coeffs& lhs, const DWT2D::Coeffs& rhs);

cv::MatExpr operator<=(const CoeffsExpr& lhs, const DWT2D::Coeffs& rhs);
cv::MatExpr operator<=(const DWT2D::Coeffs& lhs, const CoeffsExpr& rhs);

cv::MatExpr operator<=(const CoeffsExpr& lhs, const cv::MatExpr& rhs);
cv::MatExpr operator<=(const cv::MatExpr& lhs, const CoeffsExpr& rhs);

cv::MatExpr operator<=(const CoeffsExpr& lhs, const cv::Mat& rhs);
cv::MatExpr operator<=(const cv::Mat& lhs, const CoeffsExpr& rhs);

template <typename T, int m, int n>
cv::MatExpr operator<=(const CoeffsExpr& lhs, const cv::Matx<T, m, n>& rhs)
{
    cv::MatExpr compare_expression;
    internal::CompareOp::make_expression(
        compare_expression,
        cv::CMP_LE,
        lhs.coeffs,
        cv::Mat(rhs),
        is_scalar_for_array(rhs, lhs.coeffs) ? internal::CompareOp::B_IS_SCALAR : 0
    );
    return compare_expression;
}
template <typename T, int m, int n>
cv::MatExpr operator<=(const cv::Matx<T, m, n>& lhs, const CoeffsExpr& rhs)
{
    return rhs >= lhs;
}

cv::MatExpr operator<=(const CoeffsExpr& lhs, double rhs);
cv::MatExpr operator<=(double lhs, const CoeffsExpr& rhs);

cv::MatExpr operator<=(const DWT2D::Coeffs& lhs, const cv::MatExpr& rhs);
cv::MatExpr operator<=(const cv::MatExpr& lhs, const DWT2D::Coeffs& rhs);

cv::MatExpr operator<=(const DWT2D::Coeffs& lhs, const cv::Mat& rhs);
cv::MatExpr operator<=(const cv::Mat& lhs, const DWT2D::Coeffs& rhs);

template <typename T, int m, int n>
cv::MatExpr operator<=(const DWT2D::Coeffs& lhs, const cv::Matx<T, m, n>& rhs)
{
    cv::MatExpr compare_expression;
    internal::CompareOp::make_expression(
        compare_expression,
        cv::CMP_LE,
        lhs,
        cv::Mat(rhs),
        is_scalar_for_array(rhs, lhs) ? internal::CompareOp::B_IS_SCALAR : 0
    );
    return compare_expression;
}
template <typename T, int m, int n>
cv::MatExpr operator<=(const cv::Matx<T, m, n>& lhs, const DWT2D::Coeffs& rhs)
{
    return rhs >= lhs;
}

cv::MatExpr operator<=(const DWT2D::Coeffs& lhs, double rhs);
cv::MatExpr operator<=(double lhs, const DWT2D::Coeffs& rhs);

//  ----------------------------------------------------------------------------
//  greater than
cv::MatExpr operator>(const CoeffsExpr& lhs, const CoeffsExpr& rhs);

cv::MatExpr operator>(const DWT2D::Coeffs& lhs, const DWT2D::Coeffs& rhs);

cv::MatExpr operator>(const CoeffsExpr& lhs, const DWT2D::Coeffs& rhs);
cv::MatExpr operator>(const DWT2D::Coeffs& lhs, const CoeffsExpr& rhs);

cv::MatExpr operator>(const CoeffsExpr& lhs, const cv::MatExpr& rhs);
cv::MatExpr operator>(const cv::MatExpr& lhs, const CoeffsExpr& rhs);

cv::MatExpr operator>(const CoeffsExpr& lhs, const cv::Mat& rhs);
cv::MatExpr operator>(const cv::Mat& lhs, const CoeffsExpr& rhs);

template <typename T, int m, int n>
cv::MatExpr operator>(const CoeffsExpr& lhs, const cv::Matx<T, m, n>& rhs)
{
    cv::MatExpr compare_expression;
    internal::CompareOp::make_expression(
        compare_expression,
        cv::CMP_GT,
        lhs.coeffs,
        cv::Mat(rhs),
        is_scalar_for_array(rhs, lhs.coeffs) ? internal::CompareOp::B_IS_SCALAR : 0
    );
    return compare_expression;
}
template <typename T, int m, int n>
cv::MatExpr operator>(const cv::Matx<T, m, n>& lhs, const CoeffsExpr& rhs)
{
    return rhs < lhs;
}

cv::MatExpr operator>(const CoeffsExpr& lhs, double rhs);
cv::MatExpr operator>(double lhs, const CoeffsExpr& rhs);

cv::MatExpr operator>(const DWT2D::Coeffs& lhs, const cv::MatExpr& rhs);
cv::MatExpr operator>(const cv::MatExpr& lhs, const DWT2D::Coeffs& rhs);

cv::MatExpr operator>(const DWT2D::Coeffs& lhs, const cv::Mat& rhs);
cv::MatExpr operator>(const cv::Mat& lhs, const DWT2D::Coeffs& rhs);

template <typename T, int m, int n>
cv::MatExpr operator>(const DWT2D::Coeffs& lhs, const cv::Matx<T, m, n>& rhs)
{
    cv::MatExpr compare_expression;
    internal::CompareOp::make_expression(
        compare_expression,
        cv::CMP_GT,
        lhs,
        cv::Mat(rhs),
        is_scalar_for_array(rhs, lhs) ? internal::CompareOp::B_IS_SCALAR : 0
    );
    return compare_expression;
}
template <typename T, int m, int n>
cv::MatExpr operator>(const cv::Matx<T, m, n>& lhs, const DWT2D::Coeffs& rhs)
{
    return rhs < lhs;
}

cv::MatExpr operator>(const DWT2D::Coeffs& lhs, double rhs);
cv::MatExpr operator>(double lhs, const DWT2D::Coeffs& rhs);

//  ----------------------------------------------------------------------------
//  greater than or equal
cv::MatExpr operator>=(const CoeffsExpr& lhs, const CoeffsExpr& rhs);

cv::MatExpr operator>=(const DWT2D::Coeffs& lhs, const DWT2D::Coeffs& rhs);

cv::MatExpr operator>=(const CoeffsExpr& lhs, const DWT2D::Coeffs& rhs);
cv::MatExpr operator>=(const DWT2D::Coeffs& lhs, const CoeffsExpr& rhs);

cv::MatExpr operator>=(const CoeffsExpr& lhs, const cv::MatExpr& rhs);
cv::MatExpr operator>=(const cv::MatExpr& lhs, const CoeffsExpr& rhs);

cv::MatExpr operator>=(const CoeffsExpr& lhs, const cv::Mat& rhs);
cv::MatExpr operator>=(const cv::Mat& lhs, const CoeffsExpr& rhs);

template <typename T, int m, int n>
cv::MatExpr operator>=(const CoeffsExpr& lhs, const cv::Matx<T, m, n>& rhs)
{
    cv::MatExpr compare_expression;
    internal::CompareOp::make_expression(
        compare_expression,
        cv::CMP_GE,
        lhs.coeffs,
        cv::Mat(rhs),
        is_scalar_for_array(rhs, lhs.coeffs) ? internal::CompareOp::B_IS_SCALAR : 0
    );
    return compare_expression;
}
template <typename T, int m, int n>
cv::MatExpr operator>=(const cv::Matx<T, m, n>& lhs, const CoeffsExpr& rhs)
{
    return rhs <= lhs;
}

cv::MatExpr operator>=(const CoeffsExpr& lhs, double rhs);
cv::MatExpr operator>=(double lhs, const CoeffsExpr& rhs);

cv::MatExpr operator>=(const DWT2D::Coeffs& lhs, const cv::MatExpr& rhs);
cv::MatExpr operator>=(const cv::MatExpr& lhs, const DWT2D::Coeffs& rhs);

cv::MatExpr operator>=(const DWT2D::Coeffs& lhs, const cv::Mat& rhs);
cv::MatExpr operator>=(const cv::Mat& lhs, const DWT2D::Coeffs& rhs);

template <typename T, int m, int n>
cv::MatExpr operator>=(const DWT2D::Coeffs& lhs, const cv::Matx<T, m, n>& rhs)
{
    cv::MatExpr compare_expression;
    internal::CompareOp::make_expression(
        compare_expression,
        cv::CMP_GE,
        lhs,
        cv::Mat(rhs),
        is_scalar_for_array(rhs, lhs) ? internal::CompareOp::B_IS_SCALAR : 0
    );
    return compare_expression;
}
template <typename T, int m, int n>
cv::MatExpr operator>=(const cv::Matx<T, m, n>& lhs, const DWT2D::Coeffs& rhs)
{
    return rhs <= lhs;
}

cv::MatExpr operator>=(const DWT2D::Coeffs& lhs, double rhs);
cv::MatExpr operator>=(double lhs, const DWT2D::Coeffs& rhs);
} // namespace cvwt


//  ============================================================================
//  Abs, Min, Max
//  ============================================================================
namespace cv
{
//  ----------------------------------------------------------------------------
//  abs
cvwt::CoeffsExpr abs(const cvwt::CoeffsExpr& expression);
cvwt::CoeffsExpr abs(const cvwt::DWT2D::Coeffs& coeffs);

//  ----------------------------------------------------------------------------
//  max
cvwt::CoeffsExpr max(const cvwt::CoeffsExpr& a, const cvwt::CoeffsExpr& b);

cvwt::CoeffsExpr max(const cvwt::DWT2D::Coeffs& a, const cvwt::DWT2D::Coeffs& b);

cvwt::CoeffsExpr max(const cvwt::CoeffsExpr& a, const cvwt::DWT2D::Coeffs& b);
inline cvwt::CoeffsExpr max(const cvwt::DWT2D::Coeffs& a, const cvwt::CoeffsExpr& b)
{
    return max(b, a);
}

cvwt::CoeffsExpr max(const cvwt::CoeffsExpr& a, const cv::Mat& b);
inline cvwt::CoeffsExpr max(const cv::Mat& a, const cvwt::CoeffsExpr& b)
{
    return max(b, a);
}

cvwt::CoeffsExpr max(const cvwt::CoeffsExpr& a, const cv::MatExpr& b);
inline cvwt::CoeffsExpr max(const cv::MatExpr& a, const cvwt::CoeffsExpr& b)
{
    return max(b, a);
}

template <typename T, int m, int n>
cvwt::CoeffsExpr max(const cvwt::CoeffsExpr& a, const Matx<T, m, n>& b)
{
    return cvwt::CoeffsExpr(a.coeffs, max(static_cast<const MatExpr&>(a), b));
}
template <typename T, int m, int n>
cvwt::CoeffsExpr max(const Matx<T, m, n>& a, const cvwt::CoeffsExpr& b)
{
    return max(b, a);
}

cvwt::CoeffsExpr max(const cvwt::CoeffsExpr& a, double b);
inline cvwt::CoeffsExpr max(double a, const cvwt::CoeffsExpr& b)
{
    return max(b, a);
}

cvwt::CoeffsExpr max(const cvwt::DWT2D::Coeffs& a, const Mat& b);
inline cvwt::CoeffsExpr max(const Mat& a, const cvwt::DWT2D::Coeffs& b)
{
    return max(b, a);
}

template <typename T, int m, int n>
cvwt::CoeffsExpr max(const cvwt::DWT2D::Coeffs& coeffs, const Matx<T, m, n>& matrix)
{
    return cvwt::CoeffsExpr(coeffs, max(static_cast<const Mat&>(coeffs), matrix));
}
template <typename T, int m, int n>
cvwt::CoeffsExpr max(const Matx<T, m, n>& a, const cvwt::DWT2D::Coeffs& b)
{
    return max(b, a);
}

cvwt::CoeffsExpr max(const cvwt::DWT2D::Coeffs& a, const MatExpr& b);
inline cvwt::CoeffsExpr max(const MatExpr& a, const cvwt::DWT2D::Coeffs& b)
{
    return max(b, a);
}

cvwt::CoeffsExpr max(const cvwt::DWT2D::Coeffs& a, double b);
inline cvwt::CoeffsExpr max(double a, const cvwt::DWT2D::Coeffs& b)
{
    return max(b, a);
}

//  ----------------------------------------------------------------------------
//  min
cvwt::CoeffsExpr min(const cvwt::CoeffsExpr& a, const cvwt::CoeffsExpr& b);

cvwt::CoeffsExpr min(const cvwt::DWT2D::Coeffs& a, const cvwt::DWT2D::Coeffs& b);

cvwt::CoeffsExpr min(const cvwt::CoeffsExpr& a, const cvwt::DWT2D::Coeffs& b);
inline cvwt::CoeffsExpr min(const cvwt::DWT2D::Coeffs& a, const cvwt::CoeffsExpr& b)
{
    return min(b, a);
}

cvwt::CoeffsExpr min(const cvwt::CoeffsExpr& a, const cv::Mat& b);
inline cvwt::CoeffsExpr min(const cv::Mat& a, const cvwt::CoeffsExpr& b)
{
    return min(b, a);
}

cvwt::CoeffsExpr min(const cvwt::CoeffsExpr& a, const cv::MatExpr& b);
inline cvwt::CoeffsExpr min(const cv::MatExpr& a, const cvwt::CoeffsExpr& b)
{
    return min(b, a);
}

template <typename T, int m, int n>
cvwt::CoeffsExpr min(const cvwt::CoeffsExpr& a, const Matx<T, m, n>& b)
{
    return cvwt::CoeffsExpr(a.coeffs, min(static_cast<const MatExpr&>(a), b));
}
template <typename T, int m, int n>
cvwt::CoeffsExpr min(const Matx<T, m, n>& a, const cvwt::CoeffsExpr& b)
{
    return min(b, a);
}

cvwt::CoeffsExpr min(const cvwt::CoeffsExpr& a, double b);
inline cvwt::CoeffsExpr min(double a, const cvwt::CoeffsExpr& b)
{
    return min(b, a);
}

cvwt::CoeffsExpr min(const cvwt::DWT2D::Coeffs& a, const Mat& b);
inline cvwt::CoeffsExpr min(const Mat& a, const cvwt::DWT2D::Coeffs& b)
{
    return min(b, a);
}

template <typename T, int m, int n>
cvwt::CoeffsExpr min(const cvwt::DWT2D::Coeffs& coeffs, const Matx<T, m, n>& matrix)
{
    return cvwt::CoeffsExpr(coeffs, min(static_cast<const Mat&>(coeffs), matrix));
}
template <typename T, int m, int n>
cvwt::CoeffsExpr min(const Matx<T, m, n>& a, const cvwt::DWT2D::Coeffs& b)
{
    return min(b, a);
}

cvwt::CoeffsExpr min(const cvwt::DWT2D::Coeffs& a, const MatExpr& b);
inline cvwt::CoeffsExpr min(const MatExpr& a, const cvwt::DWT2D::Coeffs& b)
{
    return min(b, a);
}

cvwt::CoeffsExpr min(const cvwt::DWT2D::Coeffs& a, double b);
inline cvwt::CoeffsExpr min(double a, const cvwt::DWT2D::Coeffs& b)
{
    return min(b, a);
}
} // namespace cv

#endif  // CVWT_DWT2D_HPP

