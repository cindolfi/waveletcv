#ifndef CVWT_DWT2D_HPP
#define CVWT_DWT2D_HPP

#include "cvwt/wavelet.hpp"
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <memory>

namespace cvwt
{
/** \addtogroup dwt2d Discrete Wavelet Transform
 *  @{
 */
/**
 * @brief The DWT2D detail subbands
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

public:
    cv::Mat coeff_matrix;
    int levels;
    cv::Size image_size;
    std::vector<cv::Rect> diagonal_subband_rects;
    Wavelet wavelet;
    cv::BorderTypes border_type;
};
} // namespace internal



/**
 * @brief A two dimensional discrete wavelet transform
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
     * The coefficients at each decomposition level are comprised of three submatrices:
     * the horizontal detail subband, the vertical detail subband, and the horizontal detail subband.
     * There is a single submatrix of approximation coefficients stored alongside the coarsest details.
     * Smaller level indices correspond to smaller scales (i.e. higher resolution).
     * The submatrices are layed out as (a 3-level decomposition is shown for illustration)
     *
     * <div class="coeffs-layout-table">
     * <table>
     *     <tr>
     *         <td style="width: 12.5%; height: 12.5%">A</td>
     *         <td style="width: 12.5%; height: 12.5%">V2</td>
     *         <td rowspan="2" colspan="1" style="width: 25%; height: 25%">V1</td>
     *         <td rowspan="3" colspan="1" style="width: 50%; height: 50%">V0</td>
     *     </tr>
     *     <tr>
     *         <td style="width: 12.5%; height: 12.5%">H2</td>
     *         <td style="width: 12.5%; height: 12.5%">D2</td>
     *     </tr>
     *     <tr>
     *         <td rowspan="1" colspan="2" style="width: 25%; height: 25%">H1</td>
     *         <td rowspan="1" colspan="1" style="width: 25%; height: 25%">D1</td>
     *     </tr>
     *     <tr>
     *         <td rowspan="1" colspan="3" style="width: 50%; height: 50%">H0</td>
     *         <td rowspan="1" colspan="1" style="width: 50%; height: 50%">D0</td>
     *     </tr>
     * </table>
     * </div>
     *
     * The regions labeled H0, V0, and D0 are the level 0 (i.e. finest)
     * horizontal, vertical, and diagonal detail subbands, respectively.
     * Likewise, H1, V1, and D1 are the level 1 detail coefficients and
     * H2, V2, and D2 are the level 2 (i.e. coarsest) detail coefficients.
     * The approximation coefficients are labeled A.
     *
     * DWT2D::Coeffs objects are not constructed directly.
     * They are either returned by DWT2D::decompose() or created with DWT2D::create_coeffs().
     * The latter method is less common and is only used when algorithmically
     * generating the coefficients.
     * The only caveat is the use of the default constructor when
     * passing DWT2D::Coeffs as output parameters.
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
     * of the coefficients.  **This means that DWT2D::Coeffs can be passed to any
     * OpenCV function and acted on like a normal cv::Mat.**
     *
     * Besides the coefficients matrix, a DWT2D::Coeffs object contains
     * additional metadata.
     * In cases where an empty DWT2D::Coeffs object is needed it should be
     * created from an existing DWT2D::Coeffs with
     * empty_clone() so that the metadata retained.
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
     * In such cases the matrix can be assigned to an empty clone of the original DWT2D::Coeffs
     * @code{cpp}
     * // Using empty_clone() ensures metadata is retained.
     * auto new_coeffs = coeffs.empty_clone();
     * new_coeffs = matrix;
     * @endcode
     * or they can be assigned to original DWT2D::Coeffs
     * @code{cpp}
     * coeffs = matrix;
     * @endcode
     */
    class Coeffs
    {
        friend class DWT2D;

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
         * @param matrix
         * @param levels
         * @param image_size
         * @param subband_sizes
         * @param wavelet
         * @param border_type
         */
        Coeffs(
            const cv::Mat& matrix,
            int levels,
            const cv::Size& image_size,
            const std::vector<cv::Size>& subband_sizes,
            const Wavelet& wavelet,
            cv::BorderTypes border_type
        );
        /**
         * @brief Construct a new Coeffs object.
         *
         * @param matrix
         * @param levels
         * @param image_size
         * @param diagonal_subband_rects
         * @param wavelet
         * @param border_type
         */
        Coeffs(
            const cv::Mat& matrix,
            int levels,
            const cv::Size& image_size,
            const std::vector<cv::Rect>& diagonal_subband_rects,
            const Wavelet& wavelet,
            cv::BorderTypes border_type
        );
        /**
         * @brief Reset the coefficient metadata.
         *
         * @param size
         * @param type
         * @param levels
         * @param image_size
         * @param subband_sizes
         * @param wavelet
         * @param border_type
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
         * @brief Copy Constructor
         */
        Coeffs(const Coeffs& other) = default;
        /**
         * @brief Move Constructor
         */
        Coeffs(Coeffs&& other) = default;

        //  Assignment
        /**
         * @brief Copy assignment
         */
        Coeffs& operator=(const Coeffs& coeffs) = default;
        /**
         * @brief Move assignment
         */
        Coeffs& operator=(Coeffs&& coeffs) = default;

        /**
         * @brief Copy assignment from a cv::Mat.
         *
         * Copies the matrix to this.
         * Copy semantics is necessary to maintain a view onto the same underlying matrix.
         *
         * @warning DWT2D::Coeffs and cv::Mat have different assigment semantics.
         *      DWT2D::Coeffs **copies the values** from the matrix.
         *      cv::Mat assignment does **not copy the values** from the other matrix,
         *      it copies the shared pointer to the underlying data.
         *
         * @param matrix
         */
        Coeffs& operator=(const cv::Mat& matrix);

        /**
         * @brief Assignment from a cv::MatExpr.
         *
         * @param matrix
         */
        Coeffs& operator=(const cv::MatExpr& matrix);

        /**
         * @brief Assign all coefficients to a scalar.
         *
         * @param scalar
         */
        Coeffs& operator=(const cv::Scalar& scalar);

        //  Casting
        /**
         * @brief Implicit conversion to cv::Mat
         *
         * All metadata is discarded (e.g. levels(), wavelet(), border_type(), image_size(), etc.).
         * As such, it is impossible to cast the result of this operation back to this DWT2D::Coeffs object.
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
         * @brief Implicit cast to cv::InputArray
         *
         * This operation is implemented for interoperability with OpenCV functions.
         * Users should never need to use it directly.
         *
         * This operation is equivalent to:
         * @code{cpp}
         * cv::_InputArray(cv::Mat(*this));
         * @endcode
         *
         * @warning Modifying the elements of the returned cv::_InputArray
         *          will modify the coefficients stored by this object.
         *
         * @see `DWT2D::Coeffs::operator cv::Mat()`
         */
        operator cv::_InputArray() const { return _p->coeff_matrix; }
        /**
         * @brief Implicit cast to cv::OutputArray
         *
         * This operation is implemented for interoperability with OpenCV functions.
         * Users should never need to use it directly.
         *
         * This operation is equivalent to:
         * @code{cpp}
         * cv::_OutputArray(cv::Mat(*this));
         * @endcode
         *
         * @warning Modifying the elements of the returned cv::_OutputArray
         *          will modify the coefficients stored by this object.
         *
         * @see `DWT2D::Coeffs::operator cv::Mat()`
         */
        operator cv::_OutputArray() const { return _p->coeff_matrix; }
        /**
         * @brief Implicit cast to cv::InputOutputArray
         *
         * This operation is implemented for interoperability with OpenCV functions.
         * Users should never need to use it directly.
         *
         * This operation is equivalent to:
         * @code{cpp}
         * cv::_InputOutputArray(cv::Mat(*this));
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
         * @brief Copy
         *
         */
        Coeffs clone() const;
        /**
         * @brief Copy metadata
         *
         */
        Coeffs empty_clone() const;

        //  --------------------------------------------------------------------
        //  Get & Set Sub-coefficients
        /**
         * @brief Returns the coefficients at a given level
         *
         * @param level
         */
        Coeffs at_level(int level) const;

        /**
         * @brief
         *
         * @param level
         * @param coeffs
         */
        void set_level(int level, const cv::Mat& coeffs)
        {
            throw_if_wrong_size_for_set_level(coeffs, level);
            convert_and_copy(coeffs, _p->coeff_matrix(level_rect(level)));
        }

        /**
         * @brief
         *
         * @param level
         * @param scalar
         */
        void set_level(int level, const cv::Scalar& scalar) { _p->coeff_matrix(level_rect(level)) = scalar; }

        //  --------------------------------------------------------------------
        //  Get & Set Approximation Coefficients
        ///@{
        /**
         * @brief Get the approximation coefficients
         *
         */
        cv::Mat approx() const
        {
            return _p->coeff_matrix(approx_rect());
        }

        /**
         * @brief Set the approximation coefficients
         *
         * @param coeffs
         */
        void set_approx(const cv::Mat& coeffs)
        {
            throw_if_wrong_size_for_set_approx(coeffs);
            convert_and_copy(coeffs, approx());
        }

        /**
         * @brief Set all of the approximation coefficients to a scalar
         *
         * @param scalar
         */
        void set_approx(const cv::Scalar& scalar) { approx() = scalar; }
        ///@}

        //  --------------------------------------------------------------------
        //  Get & Set Detail Coefficients (via parameter)
        ///@{
        /**
         * @brief Get the detail coefficients at a given level and subband
         *
         * @param level
         * @param subband
         */
        cv::Mat detail(int level, int subband) const;

        /**
         * @brief Get the smallest scale detail coefficients in a given subband
         *
         * @param subband
         */
        cv::Mat detail(int subband) const { return detail(0, subband); }

        void set_all_detail_levels(cv::InputArray coeffs)
        {
            throw_if_wrong_size_for_set_all_detail_levels(coeffs);
            setTo(coeffs, detail_mask());
        }

        /**
         * @brief Set the detail coefficients at a given level and subband
         *
         * @param level The scale level
         * @param subband The detail subband - must be Subband::HORIZONTAL, Subband::VERTICAL, or Subband::DIAGONAL
         * @param coeffs The detail coefficients - must have the same size as `DWT2D::Coeffs::detail_size(level)`
         */
        void set_detail(int level, int subband, const cv::Mat& coeffs)
        {
            throw_if_wrong_size_for_set_detail(coeffs, level, subband);
            throw_if_invalid_subband(subband);
            convert_and_copy(coeffs, detail(level, subband));
        }

        /**
         * @brief Set the detail coefficients at a given level and subband
         *
         * This is an overloaded member function, provided for convenience.
         * It differs from DWT2D::Coeffs::set_detail(int level, int subband, const cv::Mat& coeffs) only in what argument(s) it accepts.
         *
         * @param level
         * @param subband
         * @param coeffs
         */
        void set_detail(int level, int subband, const cv::MatExpr& coeffs) { set_detail(level, subband, cv::Mat(coeffs)); }

        /**
         * @brief Set all detail coefficients at a given level and subband to a scalar
         *
         * @param level
         * @param subband
         * @param scalar
         */
        void set_detail(int level, int subband, const cv::Scalar& scalar) { detail(level, subband) = scalar; }

        /**
         * @brief Set the smallest scale detail coefficients in a given subband
         *
         * @param subband
         * @param coeffs
         */
        void set_detail(int subband, const cv::Mat& coeffs) { set_detail(0, subband, coeffs); }

        /**
         * @brief Set the smallest scale detail coefficients in a given subband
         *
         * This is an overloaded member function, provided for convenience.
         * It differs from DWT2D::Coeffs::set_detail(int subband, const cv::Mat& coeffs) only in what argument(s) it accepts.
         *
         * @param subband
         * @param coeffs
         */
        void set_detail(int subband, const cv::MatExpr& coeffs) { set_detail(subband, cv::Mat(coeffs)); }

        /**
         * @brief Set all of the smallest scale detail coefficients in a given subband to a scalar
         *
         * @param subband
         * @param scalar
         */
        void set_detail(int subband, const cv::Scalar& scalar) { set_detail(0, subband, scalar); }

        /**
         * @brief Returns a collection of detail coefficients at each level in a given subband
         *
         * This is equivalent to:
         * @code{cpp}
         * DWT2D::Coeffs coeffs = ...;
         * std::vector<cv::Mat> collected_details(coeffs.levels());
         * for (int level = 0; level < coeffs.levels(); ++level)
         *     collected_details[level] = coeffs.detail(level, subband);
         * @endcode
         *
         * @see
         *  - DWT2D::Coeffs::collect_horizontal_details()
         *  - DWT2D::Coeffs::collect_vertical_details()
         *  - DWT2D::Coeffs::collect_diagonal_details()
         *
         * @param subband
         */
        std::vector<cv::Mat> collect_details(int subband) const;
        ///@}

        //  --------------------------------------------------------------------
        //  Get & Set Horizontal Detail Coefficients
        ///@{
        /**
         * @brief Get the horizontal subband detail coefficients at a given level
         *
         * @param level
         */
        cv::Mat horizontal_detail(int level) const
        {
            return _p->coeff_matrix(horizontal_detail_rect(level));
        }

        /**
         * @brief Get the smallest scale horizontal subband detail coefficients
         *
         * This overload is useful when iterating over decomposition levels using a range based for loop.
         * @code{cpp}
         * DWT2D::Coeffs coeffs = ...;
         * // Get the horizontal subband detail coefficients at progressively coarser scales
         * for (auto& level_coeffs : coeffs)
         *     level_coeffs.horizontal_detail();
         * @endcode
         *
         */
        cv::Mat horizontal_detail() const { return horizontal_detail(0); }

        /**
         * @brief Set the horizontal subband detail coefficients at a given level
         *
         * @param level
         * @param coeffs
         */
        void set_horizontal_detail(int level, const cv::Mat& coeffs)
        {
            throw_if_wrong_size_for_set_detail(coeffs, level, HORIZONTAL);
            convert_and_copy(coeffs, horizontal_detail(level));
        }

        /**
         * @brief Set the horizontal subband detail coefficients at a given level
         *
         * This is an overloaded member function, provided for convenience.
         * It differs from DWT2D::Coeffs::set_horizontal_detail(int level, const cv::Mat& coeffs) only in the argument it accepts.
         *
         * @param level
         * @param coeffs
         */
        void set_horizontal_detail(int level, const cv::MatExpr& coeffs) { set_horizontal_detail(level, cv::Mat(coeffs)); }

        /**
         * @brief Set all of the horizontal subband detail coefficients at a given level to a scalar
         *
         * @param level
         * @param scalar
         */
        void set_horizontal_detail(int level, const cv::Scalar& scalar) { horizontal_detail(level) = scalar; }

        /**
         * @brief Set the smallest scale horizontal subband detail coefficients
         *
         * @param coeffs
         */
        void set_horizontal_detail(const cv::Mat& coeffs) { set_horizontal_detail(0, coeffs); }

        /**
         * @brief Set the smallest scale horizontal subband detail coefficients
         *
         * This is an overloaded member function, provided for convenience.
         * It differs from DWT2D::Coeffs::set_horizontal_detail(const cv::Mat& coeffs) only in the argument it accepts.
         *
         * @param coeffs
         */
        void set_horizontal_detail(const cv::MatExpr& coeffs) { set_horizontal_detail(cv::Mat(coeffs)); }

        /**
         * @brief Set all of the smallest scale horizontal subband detail coefficients to a scalar
         *
         * @param scalar
         */
        void set_horizontal_detail(const cv::Scalar& scalar) { set_horizontal_detail(0, scalar); }

        /**
         * @brief Returns a collection of horizontal subband detail coefficients at each level
         *
         * This is equivalent to:
         * @code{cpp}
         * DWT2D::Coeffs coeffs = ...;
         * std::vector<cv::Mat> collected_horizontal_details(coeffs.levels());
         * for (int level = 0; level < coeffs.levels(); ++level)
         *     collected_horizontal_details[level] = coeffs.horizontal_detail(level);
         * @endcode
         *
         * @see
         *  - DWT2D::Coeffs::collect_details()
         *  - DWT2D::Coeffs::collect_vertical_details()
         *  - DWT2D::Coeffs::collect_diagonal_details()
         *
         */
        std::vector<cv::Mat> collect_horizontal_details() const { return collect_details(HORIZONTAL); }
        ///@}

        //  --------------------------------------------------------------------
        //  Get & Set Vertical Detail Coefficients
        ///@{
        /**
         * @brief Get the vertical subband detail coefficients at a given level
         *
         * @param level
         */
        cv::Mat vertical_detail(int level) const
        {
            return _p->coeff_matrix(vertical_detail_rect(level));
        }

        /**
         * @brief Get the smallest scale vertical subband detail coefficients
         *
         * This overload is useful when iterating over decomposition levels using a range based for loop.
         * @code{cpp}
         * DWT2D::Coeffs coeffs = ...;
         * // Get the vertical subband detail coefficients at progressively coarser scales
         * for (auto& level_coeffs : coeffs)
         *     level_coeffs.vertical_detail();
         * @endcode
         *
         */
        cv::Mat vertical_detail() const { return vertical_detail(0); }

        /**
         * @brief Set the vertical subband detail coefficients at a given level
         *
         * @param level
         * @param coeffs
         */
        void set_vertical_detail(int level, const cv::Mat& coeffs)
        {
            throw_if_wrong_size_for_set_detail(coeffs, level, VERTICAL);
            convert_and_copy(coeffs, vertical_detail(level));
        }

        /**
         * @brief Set the vertical subband detail coefficients at a given level
         *
         * This is an overloaded member function, provided for convenience.
         * It differs from DWT2D::Coeffs::set_vertical_detail(int level, const cv::Mat& coeffs) only in the argument it accepts.
         *
         * @param level
         * @param coeffs
         */
        void set_vertical_detail(int level, const cv::MatExpr& coeffs) { set_vertical_detail(level, cv::Mat(coeffs)); }

        /**
         * @brief Set all of the vertical subband detail coefficients at a given level to a scalar
         *
         * @param level
         * @param scalar
         */
        void set_vertical_detail(int level, const cv::Scalar& scalar) { vertical_detail(level) = scalar; }

        /**
         * @brief Set the smallest scale vertical subband detail coefficients
         *
         * @param coeffs
         */
        void set_vertical_detail(const cv::Mat& coeffs) { set_vertical_detail(0, coeffs); }

        /**
         * @brief Set the smallest scale vertical subband detail coefficients
         *
         * This is an overloaded member function, provided for convenience.
         * It differs from DWT2D::Coeffs::set_vertical_detail(const cv::Mat& coeffs) only in the argument it accepts.
         *
         * @param coeffs
         */
        void set_vertical_detail(const cv::MatExpr& coeffs) { set_vertical_detail(cv::Mat(coeffs)); }

        /**
         * @brief Set all of the smallest scale vertical subband detail coefficients to a scalar
         *
         * @param scalar
         */
        void set_vertical_detail(const cv::Scalar& scalar) { set_vertical_detail(0, scalar); }

        /**
         * @brief Returns a collection of vertical subband detail coefficients at each level
         *
         * This is equivalent to:
         * @code{cpp}
         * DWT2D::Coeffs coeffs = ...;
         * std::vector<cv::Mat> collected_vertical_details(coeffs.levels());
         * for (int level = 0; level < coeffs.levels(); ++level)
         *     collected_vertical_details[level] = coeffs.vertical_detail(level);
         * @endcode
         *
         * @see
         *  - DWT2D::Coeffs::collect_details()
         *  - DWT2D::Coeffs::collect_horizontal_details()
         *  - DWT2D::Coeffs::collect_diagonal_details()
         *
         */
        std::vector<cv::Mat> collect_vertical_details() const { return collect_details(VERTICAL); }
        ///@}

        //  --------------------------------------------------------------------
        //  Get & Set Diagonal Detail Coefficients
        ///@{
        /**
         * @brief Get the diagonal subband detail coefficients at a given level
         *
         * @param level
         */
        cv::Mat diagonal_detail(int level) const
        {
            return _p->coeff_matrix(diagonal_detail_rect(level));
        }

        /**
         * @brief Get the smallest scale diagonal subband detail coefficients
         *
         * This overload is useful when iterating over decomposition levels using a range based for loop.
         * @code{cpp}
         * DWT2D::Coeffs coeffs = ...;
         * // Get the diagonal subband detail coefficients at progressively coarser scales
         * for (auto& level_coeffs : coeffs)
         *     level_coeffs.diagonal_detail();
         * @endcode
         *
         */
        cv::Mat diagonal_detail() const { return diagonal_detail(0); }

        /**
         * @brief Set the diagonal subband detail coefficients at a given level
         *
         * @param level
         * @param coeffs
         */
        void set_diagonal_detail(int level, const cv::Mat& coeffs)
        {
            throw_if_wrong_size_for_set_detail(coeffs, level, DIAGONAL);
            convert_and_copy(coeffs, diagonal_detail(level));
        }

        /**
         * @brief Set the diagonal subband detail coefficients at a given level
         *
         * This is an overloaded member function, provided for convenience.
         * It differs from DWT2D::Coeffs::set_diagonal_detail(int level, const cv::Mat& coeffs) only in the argument it accepts.
         *
         * @param level
         * @param coeffs
         */
        void set_diagonal_detail(int level, const cv::MatExpr& coeffs) { set_diagonal_detail(level, cv::Mat(coeffs)); }

        /**
         * @brief Set all of the diagonal subband detail coefficients at a given level to a scalar
         *
         * @param level
         * @param scalar
         */
        void set_diagonal_detail(int level, const cv::Scalar& scalar) { diagonal_detail(level) = scalar; }

        /**
         * @brief Set the smallest scale diagonal subband detail coefficients
         *
         * @param coeffs
         */
        void set_diagonal_detail(const cv::Mat& coeffs) { set_diagonal_detail(0, coeffs); }

        /**
         * @brief Set the smallest scale diagonal subband detail coefficients
         *
         * This is an overloaded member function, provided for convenience.
         * It differs from DWT2D::Coeffs::set_diagonal_detail(const cv::Mat& coeffs) only in the argument it accepts.
         *
         * @param coeffs
         */
        void set_diagonal_detail(const cv::MatExpr& coeffs) { set_diagonal_detail(cv::Mat(coeffs)); }

        /**
         * @brief Set all of the smallest scale diagonal subband detail coefficients to a scalar
         *
         * @param scalar
         */
        void set_diagonal_detail(const cv::Scalar& scalar) { set_diagonal_detail(0, scalar); }

        /**
         * @brief Returns a collection of diagonal subband detail coefficients at each level
         *
         * This is equivalent to:
         * @code{cpp}
         * DWT2D::Coeffs coeffs = ...;
         * std::vector<cv::Mat> collected_diagonal_details(coeffs.levels());
         * for (int level = 0; level < coeffs.levels(); ++level)
         *     collected_diagonal_details[level] = coeffs.diagonal_detail(level);
         * @endcode
         *
         * @see
         *  - DWT2D::Coeffs::collect_details()
         *  - DWT2D::Coeffs::collect_horizontal_details()
         *  - DWT2D::Coeffs::collect_vertical_details()
         *
         */
        std::vector<cv::Mat> collect_diagonal_details() const { return collect_details(DIAGONAL); }
        ///@}

        //  --------------------------------------------------------------------
        //  Sizes & Rects
        ///@{
        /**
         * @brief The size of the sub-coefficents starting at the given level.
         *
         * @param level
         */
        cv::Size level_size(int level) const;

        /**
         * @brief The region containing the sub-coefficients starting at the given level.
         *
         * @param level
         */
        cv::Rect level_rect(int level) const;

        /**
         * @brief The size of the each of the subbands at the given level.
         *
         * @param level
         */
        cv::Size detail_size(int level=0) const;

        /**
         * @brief The region containing the coefficients for the given level and subband.
         *
         * @param level
         * @param subband
         */
        cv::Rect detail_rect(int level, int subband) const;

        /**
         * @brief The region containing the smallest scale coefficients in the given subband.
         *
         * @param subband
         */
        cv::Rect detail_rect(int subband) const { return detail_rect(0, subband); }

        /**
         * @brief The region containing the approximation coeffcients.
         *
         */
        cv::Rect approx_rect() const;

        /**
         * @brief The region containing the horizontal subband coeffcients at the given level.
         *
         * @param level
         */
        cv::Rect horizontal_detail_rect(int level=0) const;

        /**
         * @brief The region containing the vertical subband coeffcients at the given level.
         *
         * @param level
         */
        cv::Rect vertical_detail_rect(int level=0) const;

        /**
         * @brief The region containing the diagonal subband coeffcients at the given level.
         *
         * @param level
         */
        cv::Rect diagonal_detail_rect(int level=0) const;
        ///@}

        //  --------------------------------------------------------------------
        //  Masks
        ///@{
        /**
         * @brief The mask indicating the invalid detail coefficients.
         *
         * Invalid detail coefficients are half rows or columns of zeros that
         * result from odd sized detail rects.  These are simply internal padding
         * and are not the result of an image decomposition and are not used
         * during reconstruction.
         *
         * Users should typically use detail_mask() when operating on
         * coefficients over one or more levels or subbands.
         */
        cv::Mat invalid_detail_mask() const;
        int total_valid() const;
        int total_details() const;

        /**
         * @brief The mask indicating the approximation coefficients.
         *
         */
        cv::Mat approx_mask() const;

        /**
         * @brief The mask indicating the detail coefficients.
         *
         */
        cv::Mat detail_mask() const;

        /**
         * @brief The mask indicating the detail coefficients at a level.
         *
         * @param level
         */
        cv::Mat detail_mask(int level) const;

        // /**
        //  * @brief The mask indicating the detail coefficients over a range of levels.
        //  *
        //  * @param lower_level
        //  * @param upper_level
        //  * @return cv::Mat
        //  */
        // cv::Mat detail_mask(int lower_level, int upper_level) const;

        /**
         * @brief The mask indicating the detail coefficients over a range of levels.
         *
         * @param levels
         */
        cv::Mat detail_mask(const cv::Range& levels) const;
        cv::Mat detail_mask(const cv::Range& levels, int subband) const;
        cv::Mat detail_mask(int level, int subband) const;
        cv::Mat detail_mask(int lower_level, int upper_level, int subband) const;

        /**
         * @brief The mask indicating the horizontal subband coefficients at the given level.
         *
         * @param level
         */
        cv::Mat horizontal_detail_mask(int level) const;
        cv::Mat horizontal_detail_mask(const cv::Range& levels) const;
        cv::Mat horizontal_detail_mask(int lower_level, int upper_level) const;
        cv::Mat horizontal_detail_mask() const { return horizontal_detail_mask(0); }

        /**
         * @brief The mask indicating the vertical subband coefficients at the given level.
         *
         * @param level
         */
        cv::Mat vertical_detail_mask(int level) const;
        cv::Mat vertical_detail_mask(const cv::Range& levels) const;
        cv::Mat vertical_detail_mask(int lower_level, int upper_level) const;
        cv::Mat vertical_detail_mask() const { return vertical_detail_mask(0); }

        /**
         * @brief The mask indicating the diagonal subband coefficients at the given level.
         *
         * @param level
         */
        cv::Mat diagonal_detail_mask(int level) const;
        cv::Mat diagonal_detail_mask(const cv::Range& levels) const;
        cv::Mat diagonal_detail_mask(int lower_level, int upper_level) const;
        cv::Mat diagonal_detail_mask() const { return diagonal_detail_mask(0); }
        ///@}

        //  --------------------------------------------------------------------
        //  Convenience cv::Mat Wrappers
        ///@{
        /**
         * @brief The number of rows.
         *
         */
        int rows() const { return _p->coeff_matrix.rows; }

        /**
         * @brief The number of columns.
         *
         */
        int cols() const { return _p->coeff_matrix.cols; }

        /**
         * @brief Returns the size of a coefficients matrix.
         *
         */
        cv::Size size() const { return _p->coeff_matrix.size(); }

        /**
         * @brief Returns the type of a coefficient.
         *
         * This is a convenience wrapper around cv::Mat::type().
         *
         */
        int type() const { return _p->coeff_matrix.type(); }

        /**
         * @brief Returns the depth of a coefficient.
         *
         * This is a convenience wrapper around cv::Mat::depth().
         *
         */
        int depth() const { return _p->coeff_matrix.depth(); }

        /**
         * @brief Returns the number of matrix channels.
         *
         * This is a convenience wrapper around cv::Mat::channels().
         *
         */
        int channels() const { return _p->coeff_matrix.channels(); }

        /**
         * @brief Returns true if the coefficients matrix has no elements.
         *
         * This is a convenience wrapper around cv::Mat::empty().
         *
         */
        bool empty() const { return _p->coeff_matrix.empty(); }

        /**
         * @brief Returns the total number of coefficients.
         *
         * This is a convenience wrapper around cv::Mat::total().
         *
         */
        size_t total() const { return _p->coeff_matrix.total(); }

        /**
         * @brief Returns the matrix element size in bytes.
         *
         * This is a convenience wrapper around cv::Mat::elemSize().
         *
         */
        size_t elemSize() const { return _p->coeff_matrix.elemSize(); }

        /**
         * @brief Returns the size of each matrix element channel in bytes.
         *
         * This is a convenience wrapper around cv::Mat::elemSize1().
         *
         */
        size_t elemSize1() const { return _p->coeff_matrix.elemSize1(); }

        /**
         * @brief Copies the coefficients to another matrix
         *
         * This is a convenience wrapper around cv::Mat::copyTo().
         *
         * @param other
         */
        void copyTo(cv::OutputArray other) const { _p->coeff_matrix.copyTo(other); }

        /**
         * @brief Copies the coefficients to another matrix
         *
         * This is a convenience wrapper around cv::Mat::copyTo().
         *
         * @param other
         * @param mask
         */
        void copyTo(cv::OutputArray other, cv::InputArray mask) const { _p->coeff_matrix.copyTo(other, mask); }

        void setTo(cv::InputArray other, cv::InputArray mask = cv::noArray()) const { _p->coeff_matrix.setTo(other, mask); }

        /**
         * @brief Converts the coefficients to another data type with optional scaling.
         *
         * This is a convenience wrapper around cv::Mat::convertTo().
         *
         * @param other
         * @param type
         * @param alpha
         * @param beta
         */
        void convertTo(cv::OutputArray other, int type, double alpha=1.0, double beta=0.0) const { _p->coeff_matrix.convertTo(other, type, alpha, beta); }

        /**
         * @brief Returns true if the coefficient matrix is stored continuously in memory.
         *
         * This is a convenience wrapper around cv::Mat::isContinuous().
         *
         */
        bool isContinuous() const { return _p->coeff_matrix.isContinuous(); }

        /**
         * @brief Returns true if the coefficient matrix is a submatrix of another matrix.
         *
         * This is a convenience wrapper around cv::Mat::isSubmatrix().
         *
         */
        bool isSubmatrix() const { return _p->coeff_matrix.isSubmatrix(); }
        ///@}

        //  --------------------------------------------------------------------
        //  Level Iterators
        ///@{
        /**
         * @brief Iterator over decomposition levels set to the finest scale
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
         * @brief Iterator over decomposition levels set after the coarsest scale
         *
         * @see begin()
         */
        LevelIterator end() { return LevelIterator(this, levels()); }
        ConstLevelIterator end() const { return ConstLevelIterator(this, levels()); }

        /**
         * @brief Constant iterator over decomposition levels set to the finest scale
         *
         * @see begin(), end()
         */
        ConstLevelIterator cbegin() const { return ConstLevelIterator(this, 0); }
        ConstLevelIterator cbegin() { return ConstLevelIterator(this, 0); }

        /**
         * @brief Constant iterator over decomposition levels set after the coarsest scale
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
         * @brief Get the number of decomposition levels.
         *
         */
        int levels() const { return _p->levels; }

        /**
         * @brief Get the wavelet used to generate the coefficients.
         *
         */
        Wavelet wavelet() const { return _p->wavelet; }

        /**
         * @brief Get the border exptrapolation method used during decomposition.
         *
         */
        cv::BorderTypes border_type() const { return _p->border_type; }

        /**
         * @brief Get the DWT2D transformation object used to compute the coeffcients.
         *
         */
        DWT2D dwt() const;

        /**
         * @brief Get the size of the image reconstructed from the coefficients at the given level.
         *
         * @param level
         */
        cv::Size image_size(int level=0) const { return level == 0 ? _p->image_size : diagonal_detail_rect(level - 1).size(); }

        /**
         * @brief Transform from DWT space back to image space.
         *
         */
        cv::Mat reconstruct() const;

        /**
         * @brief Transform from DWT space back to image space.
         *
         * @param image
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
         * @param read_mask Indicates which coefficients are used to compute the
         *                  map parameters. This can be a single channel or
         *                  multichannel matrix with depth CV_8U.
         * @param write_mask Indicates which coefficients are mapped.
         *                   This can be a single channel or multichannel matrix
         *                   with depth CV_8U.
         */
        [[nodiscard]]
        DWT2D::Coeffs map_details_to_unit_interval(
            cv::InputArray read_mask = cv::noArray(),
            cv::InputArray write_mask = cv::noArray()
        ) const;

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
         * @param[out] normalized_coeffs
         * @param[in] read_mask Indicates which coefficients are used to compute the
         *                      map parameters. This can be a single channel or
         *                      multichannel matrix with depth CV_8U.
         * @param[in] write_mask Indicates which coefficients are mapped.
         *                       This can be a single channel or multichannel matrix
         *                       with depth CV_8U.
         */
        double map_details_to_unit_interval(
            Coeffs& normalized_coeffs,
            cv::InputArray read_mask = cv::noArray(),
            cv::InputArray write_mask = cv::noArray()
        ) const;

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
         * @param write_mask Indicates which coefficients are mapped.
         *                   This can be a single channel or multichannel matrix
         *                   with depth CV_8U.
         */
        [[nodiscard]]
        DWT2D::Coeffs map_details_from_unit_interval(
            double scale,
            cv::InputArray write_mask = cv::noArray()
        ) const;

        /**
         * @brief Returns the scaling coefficient used to map the detail
         *        coefficients into the interval [0, 1].
         *
         * \f{equation}{
         *     \alpha = \frac{1}{2 \max(|w|)}
         * \f}
         *
         * @param read_mask Indicates which coefficients are used to compute the
         *                  scale. This can be a single channel or multichannel
         *                  matrix with depth CV_8U.
         */
        double map_detail_to_unit_interval_scale(cv::InputArray read_mask = cv::noArray()) const;

        cv::Range resolve_level_range(const cv::Range& levels) const
        {
            if (levels == cv::Range::all())
                return cv::Range(0, this->levels());

            return cv::Range(resolve_level(levels.start), resolve_level(levels.end));
        }
        ///@}

        friend std::vector<Coeffs> split(const Coeffs& coeffs);
        friend Coeffs merge(const std::vector<Coeffs>& coeffs);
        friend std::ostream& operator<<(std::ostream& stream, const Coeffs& wavelet);

    protected:
        //  Argument Checkers - these can be disabled by building with cmake
        //  option CVWT_ARGUMENT_CHECKING = OFF
        #if CVWT_ARGUMENT_CHECKING_ENABLED
        void throw_if_bad_mask_for_normalize(cv::InputArray mask, const std::string mask_name) const;
        void throw_if_wrong_size_for_assignment(cv::InputArray matrix) const;
        void throw_if_wrong_size_for_set_level(const cv::Mat& matrix, int level) const;
        void throw_if_wrong_size_for_set_detail(const cv::Mat& matrix, int level, int subband) const;
        void throw_if_wrong_size_for_set_all_detail_levels(cv::InputArray matrix) const;
        void throw_if_wrong_size_for_set_approx(const cv::Mat& matrix) const;
        // void throw_if_level_out_of_range(int level, const std::string& level_name = "level") const;
        void throw_if_level_out_of_range(int level) const;
        void throw_if_levels_out_of_range(int lower_level, int upper_level) const;
        void throw_if_this_is_empty() const;
        void throw_if_invalid_subband(int subband) const;
        #else
        void throw_if_bad_mask_for_normalize(cv::InputArray mask, const std::string mask_name) const noexcept {}
        void throw_if_wrong_size_for_assignment(cv::InputArray matrix) const noexcept {}
        void throw_if_wrong_size_for_set_level(const cv::Mat& matrix, int level) const noexcept {}
        void throw_if_wrong_size_for_set_detail(const cv::Mat& matrix, int level, int subband) const noexcept {}
        void throw_if_wrong_size_for_set_all_detail_levels(cv::InputArray matrix) const noexcept {};
        void throw_if_wrong_size_for_set_approx(const cv::Mat& matrix) const noexcept {}
        // void throw_if_level_out_of_range(int level, const std::string& level_name = "level") const noexcept {}
        void throw_if_level_out_of_range(int level) const noexcept {}
        void throw_if_levels_out_of_range(int lower_level, int upper_level) const noexcept {}
        void throw_if_this_is_empty() const noexcept {}
        void throw_if_invalid_subband(int subband) const noexcept {}
        #endif  // CVWT_ARGUMENT_CHECKING_ENABLED

        //  Helpers
        void convert_and_copy(const cv::Mat& source, const cv::Mat& destination);
        int resolve_level(int level) const { return (level >= 0) ? level : level + levels(); }

    private:
        std::shared_ptr<internal::Dwt2dCoeffsImpl> _p;
    };

public:
    /**
     * @brief Construct a new DWT2D object.
     *
     * @param wavelet The wavelet.
     * @param border_type The border exptrapolation method.
     */
    DWT2D(const Wavelet& wavelet, cv::BorderTypes border_type=cv::BORDER_DEFAULT);
    DWT2D() = delete;
    /**
     * @brief Copy constructor
     */
    DWT2D(const DWT2D& other) = default;
    /**
     * @brief Move constructor
     */
    DWT2D(DWT2D&& other) = default;

    /**
     * @brief Alias of decompose(cv::InputArray image) const.
     *
     * @param image The image to be transformed.
     */
    Coeffs operator()(cv::InputArray image) const { return decompose(image); }
    /**
     * @brief Alias of decompose(cv::InputArray image, int levels) const.
     *
     * @param image The image to be transformed.
     * @param levels The number of levels.
     */
    Coeffs operator()(cv::InputArray image, int levels) const { return decompose(image, levels); }
    /**
     * @brief Alias of decompose(cv::InputArray image, Coeffs& output) const.
     *
     * @param image The image to be transformed.
     * @param output The result of the discrete wavelet transformation.
     */
    void operator()(cv::InputArray image, Coeffs& output) const { decompose(image, output); }
    /**
     * @brief Alias of decompose(cv::InputArray image, Coeffs& output, int levels) const.
     *
     * @param image The image to be transformed.
     * @param output The result of the discrete wavelet transformation.
     * @param levels The number of levels.
     */
    void operator()(cv::InputArray image, Coeffs& output, int levels) const { decompose(image, output, levels); }

    /**
     * @brief Perform a multiscale discrete wavelet transformation.
     *
     * @param image The image to be transformed.
     * @param output The result of the discrete wavelet transformation.
     * @param levels The number of levels.
     */
    void decompose(cv::InputArray image, Coeffs& output, int levels) const;

    /**
     * @brief Perform a multiscale discrete wavelet transformation.
     *
     * This is a overloaded member function, provided for convenience.
     * It is equivalent to:
     * @code{cpp}
     * DWT2D::Coeffs result;
     * this->decompose(image, result, this->max_levels_without_border_effects(image));
     * @endcode
     *
     * @param image The image to be transformed.
     */
    Coeffs decompose(cv::InputArray image) const
    {
        Coeffs coeffs;
        decompose(image, coeffs);
        return coeffs;
    }

    /**
     * @brief Perform a multiscale discrete wavelet transformation.
     *
     * This is a overloaded member function, provided for convenience.
     * It is equivalent to:
     * @code{cpp}
     * this->decompose(image, output, this->max_levels_without_border_effects(image));
     * @endcode
     *
     * @param image The image to be transformed.
     * @param output The result of the discrete wavelet transformation.
     */
    void decompose(cv::InputArray image, Coeffs& output) const
    {
        decompose(image, output, max_levels_without_border_effects(image));
    }

    /**
     * @brief Perform a multiscale discrete wavelet transformation.
     *
     * This is a overloaded member function, provided for convenience.
     * It is equivalent to:
     * @code{cpp}
     * DWT2D::Coeffs result;
     * this->decompose(image, result, levels);
     * @endcode
     *
     * @param image The image to be transformed.
     * @param levels
     */
    Coeffs decompose(cv::InputArray image, int levels) const
    {
        DWT2D::Coeffs coeffs;
        decompose(image, coeffs, levels);
        return coeffs;
    }

    /**
     * @brief Reconstruct an image from DWT coefficients.
     *
     * This performs an inverse multilevel discrete wavelet transformation.
     *
     * @param coeffs The discrete wavelet transform coefficients.
     * @param output The reconstructed image.
     */
    void reconstruct(const Coeffs& coeffs, cv::OutputArray output) const;
    /**
     * @brief Reconstruct an image from DWT coefficients.
     *
     * This is a overloaded member function, provided for convenience.
     * It is equivalent to:
     * @code{cpp}
     * cv::Mat result;
     * this->reconstruct(coeffs, result);
     * @endcode
     *
     * @param coeffs
     */
    cv::Mat reconstruct(const Coeffs& coeffs) const
    {
        cv::Mat output;
        reconstruct(coeffs, output);
        return output;
    }

    /**
     * @brief Create a DWT2D::Coeffs object.
     *
     * @param coeffs_matrix The initial discrete wavelet transform coefficients.
     * @param image_size The size of the reconstructed image.
     * @param levels The number of levels.
     */
    Coeffs create_coeffs(
        cv::InputArray coeffs_matrix,
        const cv::Size& image_size,
        int levels
    ) const;

    /**
     * @brief Create a zero initialized DWT2D::Coeffs object.
     *
     * @param image_size The size of the reconstructed image.
     * @param type The type of the reconstructed image.
     * @param levels The number of levels.
     */
    Coeffs create_coeffs(const cv::Size& image_size, int type, int levels) const;

    /**
     * @brief Create a zero initialized DWT2D::Coeffs object.
     *
     * This is a overloaded member function, provided for convenience.
     * It is equivalent to:
     * @code{cpp}
     * this->create_coeffs(image.size(), image.type(), levels);
     * @endcode
     *
     * @param image
     * @param levels
     */
    Coeffs create_coeffs(cv::InputArray image, int levels) const
    {
        return create_coeffs(image.size(), image.type(), levels);
    }

    /**
     * @brief Create a zero initialized DWT2D::Coeffs object.
     *
     * This is a overloaded member function, provided for convenience.
     * It is equivalent to:
     * @code{cpp}
     * this->create_coeffs(cv::Size(image_cols, image_rows), type, levels);
     * @endcode
     *
     * @param image_rows
     * @param image_cols
     * @param type
     * @param levels
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
     * @param image_size
     * @param levels
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
     * @param image
     * @param levels
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
     * @param image_rows
     * @param image_cols
     * @param levels
     */
    cv::Size coeffs_size_for_image(int image_rows, int image_cols, int levels) const
    {
        return coeffs_size_for_image(cv::Size(image_cols, image_rows), levels);
    }

    /**
     * @brief Returns the maximum number of decomposition levels possible while maintaining perfect reconstruction
     *
     * @param image_size
     */
    int max_levels_without_border_effects(const cv::Size& image_size) const
    {
        return max_levels_without_border_effects(image_size.height, image_size.width);
    }

    /**
     * @brief Returns the maximum number of decomposition levels possible while maintaining perfect reconstruction
     *
     * @param image_rows
     * @param image_cols
     */
    int max_levels_without_border_effects(int image_rows, int image_cols) const;

    /**
     * @brief Returns the maximum number of decomposition levels possible while maintaining perfect reconstruction
     *
     * @param image
     */
    int max_levels_without_border_effects(cv::InputArray image) const
    {
        return max_levels_without_border_effects(image.size());
    }
protected:
    //  Argument Checkers - these can be disabled by building with cmake
    //  option CVWT_ARGUMENT_CHECKING = OFF
    #if CVWT_ARGUMENT_CHECKING_ENABLED
    void throw_if_levels_out_of_range(int levels) const;
    void throw_if_inconsistent_coeffs_and_image_sizes(
        cv::InputArray coeffs,
        const cv::Size& image_size,
        int levels
    ) const;
    #else
    void throw_if_levels_out_of_range(int levels) const noexcept {}
    void throw_if_inconsistent_coeffs_and_image_sizes(
        cv::InputArray coeffs,
        const cv::Size& image_size,
        int levels
    ) const noexcept
    {}
    #endif  // CVWT_ARGUMENT_CHECKING_ENABLED

    //  Log warnings - these can be disabled by defining CVWT_DISABLE_DWT_WARNINGS_ENABLED
    void warn_if_border_effects_will_occur(int levels, const cv::Size& image_size) const noexcept;
    void warn_if_border_effects_will_occur(int levels, cv::InputArray image) const noexcept;
    void warn_if_border_effects_will_occur(const Coeffs& coeffs) const noexcept;

    std::vector<cv::Size> calc_subband_sizes(const cv::Size& image_size, int levels) const;
public:
    Wavelet wavelet;
    cv::BorderTypes border_type;
};

std::vector<DWT2D::Coeffs> split(const DWT2D::Coeffs& coeffs);
DWT2D::Coeffs merge(const std::vector<DWT2D::Coeffs>& coeffs);
std::ostream& operator<<(std::ostream& stream, const DWT2D::Coeffs& wavelet);


//  ----------------------------------------------------------------------------
//  Functional Interface
//  ----------------------------------------------------------------------------
/**
 * @{ DWT Functional API
 */
/**
 * @brief Perform a multiscale discrete wavelet transform.
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
 * @param image
 * @param wavelet
 * @param border_type
 */
DWT2D::Coeffs dwt2d(
    cv::InputArray image,
    const Wavelet& wavelet,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

/**
 * @brief Perform a multiscale discrete wavelet transform.
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
 * @param image
 * @param wavelet
 * @param border_type
 */
DWT2D::Coeffs dwt2d(
    cv::InputArray image,
    const std::string& wavelet,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

/**
 * @brief Decompose an image using a multiscale discrete wavelet transform.
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
 * @param image
 * @param wavelet
 * @param levels
 * @param border_type
 */
DWT2D::Coeffs dwt2d(
    cv::InputArray image,
    const Wavelet& wavelet,
    int levels,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

/**
 * @brief Decompose an image using a multiscale discrete wavelet transform.
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
 * @param image
 * @param wavelet
 * @param levels
 * @param border_type
 */
DWT2D::Coeffs dwt2d(
    cv::InputArray image,
    const std::string& wavelet,
    int levels,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

/**
 * @brief Decompose an image using a multiscale discrete wavelet transform.
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
 * @param image
 * @param output
 * @param wavelet
 * @param border_type
 */
void dwt2d(
    cv::InputArray image,
    DWT2D::Coeffs& output,
    const Wavelet& wavelet,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

/**
 * @brief Decompose an image using a multiscale discrete wavelet transform.
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
 * @param image
 * @param output
 * @param wavelet
 * @param border_type
 */
void dwt2d(
    cv::InputArray image,
    DWT2D::Coeffs& output,
    const std::string& wavelet,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);
/**
 * @brief Decompose an image using a multiscale discrete wavelet transform.
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
 * @param image
 * @param output
 * @param wavelet
 * @param levels
 * @param border_type
 */
void dwt2d(
    cv::InputArray image,
    DWT2D::Coeffs& output,
    const Wavelet& wavelet,
    int levels,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);
/**
 * @brief Decompose an image using a multiscale discrete wavelet transform.
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
 * @param image
 * @param output
 * @param wavelet
 * @param levels
 * @param border_type
 */
void dwt2d(
    cv::InputArray image,
    DWT2D::Coeffs& output,
    const std::string& wavelet,
    int levels,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

/**
 * @brief Reconstruct an image from DWT coefficients.
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
 * @param coeffs
 * @param output
 * @param wavelet
 * @param border_type
 */
void idwt2d(
    const DWT2D::Coeffs& coeffs,
    cv::OutputArray output,
    const Wavelet& wavelet,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

/**
 * @brief Reconstruct an image from DWT coefficients.
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
 * @param coeffs
 * @param output
 * @param wavelet
 * @param border_type
 */
void idwt2d(
    const DWT2D::Coeffs& coeffs,
    cv::OutputArray output,
    const std::string& wavelet,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

/**
 * @brief Reconstruct an image from DWT coefficients.
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
 * @param coeffs
 * @param wavelet
 * @param border_type
 */
cv::Mat idwt2d(
    const DWT2D::Coeffs& coeffs,
    const Wavelet& wavelet,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

/**
 * @brief Reconstruct an image from DWT coefficients.
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
 * @param coeffs
 * @param wavelet
 * @param border_type
 */
cv::Mat idwt2d(
    const DWT2D::Coeffs& coeffs,
    const std::string& wavelet,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);
/** @} DWT Functional API*/
/** @} dwt2d*/

} // namespace cvwt

#endif  // CVWT_DWT2D_HPP

