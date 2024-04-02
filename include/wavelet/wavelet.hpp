#ifndef WAVELET_WAVELET_HPP
#define WAVELET_WAVELET_HPP

#include <opencv2/core.hpp>
#include <string>
#include <memory>
#include <vector>
#include <array>
#include <map>

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
struct DecomposeKernels
{
    DecomposeKernels() :
        lowpass(0, 0, CV_64F),
        highpass(0, 0, CV_64F)
    {}

    DecomposeKernels(const cv::Mat& lowpass, const cv::Mat& highpass) :
        lowpass(lowpass),
        highpass(highpass)
    {}

    DecomposeKernels(const DecomposeKernels& other) = default;
    DecomposeKernels(DecomposeKernels&& other) = default;

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
        even_lowpass(even_lowpass),
        odd_lowpass(odd_lowpass),
        even_highpass(even_highpass),
        odd_highpass(odd_highpass)
    {}

    ReconstructKernels(const ReconstructKernels& other) = default;
    ReconstructKernels(ReconstructKernels&& other) = default;

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

    void check_kernels(
        const cv::Mat& reconstruct_lowpass,
        const cv::Mat& reconstruct_highpass,
        const cv::Mat& decompose_lowpass,
        const cv::Mat& decompose_highpass
    ) const;

    KernelPair decompose_kernels() const;
    KernelPair reconstruct_kernels() const;

    DecomposeKernels decompose;
    ReconstructKernels reconstruct;
    int filter_length;
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

    bool empty() const { return _p->decompose.lowpass.empty(); }
    int type() const { return _p->decompose.lowpass.type(); }
    int depth() const { return _p->decompose.lowpass.depth(); }
    int filter_length() const { return _p->filter_length; }
    KernelPair reconstruct_kernels() const { return _p->reconstruct_kernels(); }
    KernelPair decompose_kernels() const { return _p->decompose_kernels(); }

    bool operator==(const FilterBank& other) const;
    friend std::ostream& operator<<(std::ostream& stream, const FilterBank& filter_bank);

    void forward(
        cv::InputArray x,
        cv::OutputArray approx,
        cv::OutputArray horizontal_detail,
        cv::OutputArray vertical_detail,
        cv::OutputArray diagonal_detail,
        int border_type=cv::BORDER_DEFAULT,
        const cv::Scalar& border_value=cv::Scalar()
    ) const;
    void prepare_forward(int type) const;
    void finish_forward() const;
    bool is_prepared_forward(int type) const
    {
        return (bool)_decompose
        && _decompose->lowpass.type() == promote_type(type);
    }

    void inverse(
        cv::InputArray approx,
        cv::InputArray horizontal_detail,
        cv::InputArray vertical_detail,
        cv::InputArray diagonal_detail,
        cv::OutputArray output,
        const cv::Size& output_size
    ) const;

    void inverse(
        cv::InputArray approx,
        cv::InputArray horizontal_detail,
        cv::InputArray vertical_detail,
        cv::InputArray diagonal_detail,
        cv::OutputArray output
    ) const
    {
        inverse(
            approx,
            horizontal_detail,
            vertical_detail,
            diagonal_detail,
            output,
            cv::Size(approx.size() * 2)
        );
    }
    void prepare_inverse(int type) const;
    void finish_inverse() const;
    bool is_prepared_inverse(int type) const
    {
        return (bool)_reconstruct
        && _reconstruct->even_lowpass.type() == promote_type(type);
    }

    cv::Size output_size(const cv::Size& input_size) const;
    int output_size(int input_size) const;
    cv::Size subband_size(const cv::Size& input_size) const;
    int subband_size(int input_size) const;

    //  kernel pair factories
    template <typename T>
    static KernelPair create_orthogonal_decompose_kernels(const std::vector<T>& reconstruct_lowpass_coeffs)
    {
        return create_biorthogonal_decompose_kernels(
            reconstruct_lowpass_coeffs,
            reconstruct_lowpass_coeffs
        );
    }

    template <typename T>
    static KernelPair create_orthogonal_reconstruct_kernels(const std::vector<T>& reconstruct_lowpass_coeffs)
    {
        return create_biorthogonal_reconstruct_kernels(
            reconstruct_lowpass_coeffs,
            reconstruct_lowpass_coeffs
        );
    }

    template <typename T>
    static KernelPair create_biorthogonal_decompose_kernels(
        const std::vector<T>& reconstruct_lowpass_coeffs,
        const std::vector<T>& decompose_lowpass_coeffs
    )
    {
        std::vector<T> lowpass(decompose_lowpass_coeffs.size());
        std::ranges::reverse_copy(decompose_lowpass_coeffs, lowpass.begin());

        std::vector<T> highpass = reconstruct_lowpass_coeffs;
        negate_evens(highpass);

        return KernelPair(lowpass, highpass);
    }

    template <typename T>
    static KernelPair create_biorthogonal_reconstruct_kernels(
        const std::vector<T>& reconstruct_lowpass_coeffs,
        const std::vector<T>& decompose_lowpass_coeffs
    )
    {
        std::vector<T> lowpass = reconstruct_lowpass_coeffs;

        std::vector<T> highpass(decompose_lowpass_coeffs.size());
        std::ranges::reverse_copy(decompose_lowpass_coeffs, highpass.begin());
        negate_odds(highpass);

        return KernelPair(lowpass, highpass);
    }

    template <typename T>
    static void negate_odds(std::vector<T>& coeffs)
    {
        std::ranges::transform(
            coeffs,
            coeffs.begin(),
            [&, i = 0] (auto coeff) mutable { return (i++ % 2 ? -1 : 1) * coeff; }
        );
    }

    template <typename T>
    static void negate_evens(std::vector<T>& coeffs)
    {
        std::ranges::transform(
            coeffs,
            coeffs.begin(),
            [&, i = 0] (auto coeff) mutable { return (i++ % 2 ? 1 : -1) * coeff; }
        );
    }

    int promote_type(int type) const;
protected:
    void promote_input(
        cv::InputArray input,
        cv::OutputArray promoted_input
    ) const;
    void promote_kernel(
        cv::InputArray kernel,
        cv::OutputArray promoted_kernel,
        int type
    ) const;
    void pad(
        cv::InputArray data,
        cv::OutputArray output,
        int border_type=cv::BORDER_DEFAULT,
        const cv::Scalar& border_value=cv::Scalar()
    ) const;
    void convolve_rows_and_downsample_cols(
        cv::InputArray data,
        cv::OutputArray output,
        const cv::Mat& kernel
    ) const;
    void convolve_cols_and_downsample_rows(
        cv::InputArray data,
        cv::OutputArray output,
        const cv::Mat& kernel
    ) const;
    void upsample_cols_and_convolve_rows(
        cv::InputArray data,
        cv::OutputArray output,
        const cv::Mat& even_kernel,
        const cv::Mat& odd_kernel,
        const cv::Size& output_size
    ) const;
    void upsample_rows_and_convolve_cols(
        cv::InputArray data,
        cv::OutputArray output,
        const cv::Mat& even_kernel,
        const cv::Mat& odd_kernel,
        const cv::Size& output_size
    ) const;

    //  Argument Checkers - these all raise execeptions and can be disabled by
    //  defining DISABLE_ARG_CHECKS
    void check_forward_input(cv::InputArray input) const;
    void check_inverse_inputs(
        cv::InputArray approx,
        cv::InputArray horizontal_detail,
        cv::InputArray vertical_detail,
        cv::InputArray diagonal_detail
    ) const;

protected:
    std::shared_ptr<internal::FilterBankImpl> _p;
    //  These are used as temporary caches for promoted kernels to avoid
    //  converting kernels every time forward or inverse is called by multilevel
    //  algorithms (e.g. DWT2D).  The caching and freeing is done by the
    //  prepare_*() and finish_*() methods, respectively.
    mutable std::shared_ptr<internal::DecomposeKernels> _decompose;
    mutable std::shared_ptr<internal::ReconstructKernels> _reconstruct;
};

std::ostream& operator<<(std::ostream& stream, const FilterBank& filter_bank);


enum class Symmetry {
    SYMMETRIC,
    NEAR_SYMMETRIC,
    ASYMMETRIC,
};

class Wavelet
{
protected:
    struct WaveletImpl
    {
        int vanishing_moments_psi = 0;
        int vanishing_moments_phi = 0;
        int support_width = 0;
        bool orthogonal = false;
        bool biorthogonal = false;
        Symmetry symmetry = Symmetry::ASYMMETRIC;
        bool compact_support = true;
        std::string family;
        std::string name;
        FilterBank filter_bank;
    };

public:
    Wavelet();
    Wavelet(
        int vanishing_moments_psi,
        int vanishing_moments_phi,
        int support_width,
        bool orthogonal,
        bool biorthogonal,
        Symmetry symmetry,
        bool compact_support,
        const std::string& family,
        const std::string& name,
        const FilterBank& filter_bank
    );

    int vanishing_moments_psi() const { return _p->vanishing_moments_psi; }
    int vanishing_moments_phi() const { return _p->vanishing_moments_phi; }
    int support_width() const { return _p->support_width; }
    bool orthogonal() const { return _p->orthogonal; }
    bool biorthogonal() const { return _p->biorthogonal; }
    Symmetry symmetry() const { return _p->symmetry; }
    bool compact_support() const { return _p->compact_support; }
    std::string family() const { return _p->family; }
    std::string name() const { return _p->name; }
    FilterBank filter_bank() const { return _p->filter_bank; }
    int filter_length() const { return _p->filter_bank.filter_length(); }
    bool is_valid() const { return _p->filter_bank.filter_length() > 0; }

    static Wavelet create(const std::string& name);
    static std::vector<std::string> registered_wavelets();
    template<class... Args>
    static void register_factory(
        const std::string& name,
        Wavelet factory(Args...),
        const Args&... args
    )
    {
        _wavelet_factories[name] = std::bind(factory, args...);
    }

    bool operator==(const Wavelet& other) const;

    friend std::ostream& operator<<(std::ostream& stream, const Wavelet& wavelet);
private:
    std::shared_ptr<WaveletImpl> _p;
    static std::map<std::string, std::function<Wavelet()>> _wavelet_factories;
};

std::ostream& operator<<(std::ostream& stream, const Wavelet& wavelet);

/**
 * Factories
*/
Wavelet daubechies(int order);
Wavelet haar();
Wavelet symlets(int order);
Wavelet coiflets(int order);
Wavelet biorthogonal(int vanishing_moments_psi, int vanishing_moments_phi);

namespace internal
{
    void check_wavelet_order(int order, int min_order, int max_order, const std::string family);
    std::string get_and_check_biorthogonal_name(int vanishing_moments_psi, int vanishing_moments_phi);
} // namespace internal
} // namespace wavelet

#endif  // WAVELET_WAVELET_HPP

