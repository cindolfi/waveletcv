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
class WaveletFilterBank
{
public:
    class KernelPair
    {
    public:
        KernelPair(const cv::Mat& lowpass, const cv::Mat& highpass) :
            _lowpass(lowpass),
            _highpass(highpass)
        {}

        template <typename T>
        KernelPair(const std::vector<T>& lowpass, const std::vector<T>& highpass) :
            _lowpass(cv::Mat(lowpass, true)),
            _highpass(cv::Mat(highpass, true))
        {}

        template <typename T, int N>
        KernelPair(const std::array<T, N>& lowpass, const std::array<T, N>& highpass) :
            _lowpass(cv::Mat(lowpass, true)),
            _highpass(cv::Mat(highpass, true))
        {}

        const cv::Mat& lowpass() const { return _lowpass; }
        const cv::Mat& highpass() const { return _highpass; }
    private:
        cv::Mat _lowpass;
        cv::Mat _highpass;
    };

public:
    WaveletFilterBank(
        const KernelPair& reconstruct_kernels,
        const KernelPair& decompose_kernels
    );
    WaveletFilterBank(
        const cv::Mat& reconstruct_lowpass,
        const cv::Mat& reconstruct_highpass,
        const cv::Mat& decompose_lowpass,
        const cv::Mat& decompose_highpass
    );
    WaveletFilterBank() = default;
    WaveletFilterBank(const WaveletFilterBank& other) = default;
    WaveletFilterBank(WaveletFilterBank&& other) = default;

    int filter_length() const
    {
        return std::max(
            _decompose_kernels.lowpass().total(),
            _reconstruct_kernels.lowpass().total()
        );
    }

    cv::Size output_size(const cv::Size& input_size) const;
    int output_size(int input_size) const;
    cv::Size subband_size(const cv::Size& input_size) const;
    int subband_size(int input_size) const;

    // cv::Size input_size(const cv::Size& output_size) const;
    // int input_size(int output_size) const;
    cv::Rect unpad_rect(const cv::Size& coeffs_size) const;
    cv::Rect unpad_rect(cv::InputArray padded_matrix) const;
    void unpad(cv::InputArray padded_matrix, cv::OutputArray output) const;

    void forward(
        cv::InputArray x,
        cv::OutputArray approx,
        cv::OutputArray horizontal_detail,
        cv::OutputArray vertical_detail,
        cv::OutputArray diagonal_detail,
        int border_type=cv::BORDER_DEFAULT
    ) const;

    cv::Mat forward_stage1_lowpass(cv::InputArray data, int border_type=cv::BORDER_DEFAULT) const;
    void forward_stage1_lowpass(cv::InputArray data, cv::OutputArray output, int border_type=cv::BORDER_DEFAULT) const;

    cv::Mat forward_stage1_highpass(cv::InputArray data, int border_type=cv::BORDER_DEFAULT) const;
    void forward_stage1_highpass(cv::InputArray data, cv::OutputArray output, int border_type=cv::BORDER_DEFAULT) const;

    cv::Mat forward_stage2_lowpass(cv::InputArray data, int border_type=cv::BORDER_DEFAULT) const;
    void forward_stage2_lowpass(cv::InputArray data, cv::OutputArray output, int border_type=cv::BORDER_DEFAULT) const;

    cv::Mat forward_stage2_highpass(cv::InputArray data, int border_type=cv::BORDER_DEFAULT) const;
    void forward_stage2_highpass(cv::InputArray data, cv::OutputArray output, int border_type=cv::BORDER_DEFAULT) const;

    void inverse(
        cv::InputArray approx,
        cv::InputArray horizontal_detail,
        cv::InputArray vertical_detail,
        cv::InputArray diagonal_detail,
        cv::OutputArray output,
        int border_type=cv::BORDER_DEFAULT
    ) const;

    cv::Mat inverse_stage1_lowpass(cv::InputArray data, int border_type=cv::BORDER_DEFAULT) const;
    void inverse_stage1_lowpass(cv::InputArray data, cv::OutputArray output, int border_type=cv::BORDER_DEFAULT, int offset=0) const;

    cv::Mat inverse_stage1_highpass(cv::InputArray data, int border_type=cv::BORDER_DEFAULT) const;
    void inverse_stage1_highpass(cv::InputArray data, cv::OutputArray output, int border_type=cv::BORDER_DEFAULT, int offset=0) const;

    cv::Mat inverse_stage2_lowpass(cv::InputArray data, int border_type=cv::BORDER_DEFAULT) const;
    void inverse_stage2_lowpass(cv::InputArray data, cv::OutputArray output, int border_type=cv::BORDER_DEFAULT, int offset=0) const;

    cv::Mat inverse_stage2_highpass(cv::InputArray data, int border_type=cv::BORDER_DEFAULT) const;
    void inverse_stage2_highpass(cv::InputArray data, cv::OutputArray output, int border_type=cv::BORDER_DEFAULT, int offset=0) const;

    KernelPair reconstruct_kernels() const { return _reconstruct_kernels; }
    KernelPair decompose_kernels() const { return _decompose_kernels; }

    template <typename T>
    static KernelPair build_orthogonal_decompose_kernels(const std::vector<T>& reconstruct_lowpass_coeffs)
    {
        return build_biorthogonal_decompose_kernels(reconstruct_lowpass_coeffs, reconstruct_lowpass_coeffs);
    }

    template <typename T>
    static KernelPair build_orthogonal_reconstruct_kernels(const std::vector<T>& reconstruct_lowpass_coeffs)
    {
        return build_biorthogonal_reconstruct_kernels(reconstruct_lowpass_coeffs, reconstruct_lowpass_coeffs);
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

protected:
    void downsample_rows(cv::InputArray data, cv::OutputArray output) const;
    void downsample_cols(cv::InputArray data, cv::OutputArray output) const;
    void upsample_rows(cv::InputArray data, cv::OutputArray output) const;
    void upsample_cols(cv::InputArray data, cv::OutputArray output) const;
    void filter_rows(cv::InputArray data, cv::OutputArray output, const cv::Mat& kernel, int border_type) const;
    void filter_cols(cv::InputArray data, cv::OutputArray output, const cv::Mat& kernel, int border_type) const;

    KernelPair _reconstruct_kernels;
    KernelPair _decompose_kernels;
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
        // int order;
        int vanishing_moments_psi;
        int vanishing_moments_phi;
        int support_width;
        bool orthogonal;
        bool biorthogonal;
        Symmetry symmetry;
        bool compact_support;
        std::string family;
        std::string name;
        FilterBank filter_bank;
    };

public:
    Wavelet(
        // int order,
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
    Wavelet() = delete;

    // int order() const { return _p->order; }
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

