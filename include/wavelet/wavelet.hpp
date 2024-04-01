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
namespace internal
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
    {
    }

    template <typename T>
    KernelPair(const std::vector<T>& lowpass, const std::vector<T>& highpass) :
        KernelPair(cv::Mat(lowpass, true), cv::Mat(highpass, true))
    {}

    template <typename T, int N>
    KernelPair(const std::array<T, N>& lowpass, const std::array<T, N>& highpass) :
        KernelPair(cv::Mat(lowpass, true), cv::Mat(highpass, true))
    {}

    const cv::Mat lowpass() const { return _lowpass; }
    const cv::Mat highpass() const { return _highpass; }

    int filter_length() const { return _lowpass.total(); }
    bool empty() const { return _lowpass.empty(); }

    bool operator==(const KernelPair& other) const;

protected:
    cv::Mat _lowpass;
    cv::Mat _highpass;
};

struct WaveletFilterBankImpl
{
    WaveletFilterBankImpl();

    WaveletFilterBankImpl(
        const cv::Mat& reconstruct_lowpass,
        const cv::Mat& reconstruct_highpass,
        const cv::Mat& decompose_lowpass,
        const cv::Mat& decompose_highpass
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

    KernelPair reconstruct_kernels() const;
    KernelPair decompose_kernels() const;

    cv::Mat decompose_lowpass;
    cv::Mat decompose_highpass;
    cv::Mat reconstruct_even_lowpass;
    cv::Mat reconstruct_odd_lowpass;
    cv::Mat reconstruct_even_highpass;
    cv::Mat reconstruct_odd_highpass;
    int filter_length;
};

class WaveletFilterBank
{
public:
    WaveletFilterBank();
    WaveletFilterBank(
        const cv::Mat& reconstruct_lowpass,
        const cv::Mat& reconstruct_highpass,
        const cv::Mat& decompose_lowpass,
        const cv::Mat& decompose_highpass
    );
    WaveletFilterBank(
        const KernelPair& reconstruct_kernels,
        const KernelPair& decompose_kernels
    );
    WaveletFilterBank(const WaveletFilterBank& other) = default;
    WaveletFilterBank(WaveletFilterBank&& other) = default;

    bool empty() const { return _p->decompose_lowpass.empty(); }
    int filter_length() const { return _p->filter_length; }
    KernelPair reconstruct_kernels() const { return _p->reconstruct_kernels(); }
    KernelPair decompose_kernels() const { return _p->decompose_kernels(); }

    bool operator==(const WaveletFilterBank& other) const;

    void forward(
        cv::InputArray x,
        cv::OutputArray approx,
        cv::OutputArray horizontal_detail,
        cv::OutputArray vertical_detail,
        cv::OutputArray diagonal_detail,
        int border_type=cv::BORDER_DEFAULT,
        const cv::Scalar& border_value=cv::Scalar()
    ) const;

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

    cv::Size output_size(const cv::Size& input_size) const;
    int output_size(int input_size) const;
    cv::Size subband_size(const cv::Size& input_size) const;
    int subband_size(int input_size) const;

    //  kernel pair factories
    template <typename T>
    static KernelPair build_orthogonal_decompose_kernels(const std::vector<T>& reconstruct_lowpass_coeffs)
    {
        return build_biorthogonal_decompose_kernels(
            reconstruct_lowpass_coeffs,
            reconstruct_lowpass_coeffs
        );
    }

    template <typename T>
    static KernelPair build_orthogonal_reconstruct_kernels(const std::vector<T>& reconstruct_lowpass_coeffs)
    {
        return build_biorthogonal_reconstruct_kernels(
            reconstruct_lowpass_coeffs,
            reconstruct_lowpass_coeffs
        );
    }

    template <typename T>
    static KernelPair build_biorthogonal_decompose_kernels(
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
    static KernelPair build_biorthogonal_reconstruct_kernels(
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

    friend std::ostream& operator<<(std::ostream& stream, const WaveletFilterBank& filter_bank);

protected:
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
        const cv::Size& valid_size
    ) const;
    void upsample_rows_and_convolve_cols(
        cv::InputArray data,
        cv::OutputArray output,
        const cv::Mat& even_kernel,
        const cv::Mat& odd_kernel,
        const cv::Size& valid_size
    ) const;

protected:
    std::shared_ptr<WaveletFilterBankImpl> _p;
};

std::ostream& operator<<(std::ostream& stream, const WaveletFilterBank& filter_bank);
} // namespace internal




class Wavelet
{
public:
    using FilterBank = internal::WaveletFilterBank;
    enum class Symmetry {
        SYMMETRIC,
        NEAR_SYMMETRIC,
        ASYMMETRIC,
    };

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
    Wavelet();

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

}   // namespace wavelet

#endif  // WAVELET_WAVELET_HPP

