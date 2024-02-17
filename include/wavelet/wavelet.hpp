#ifndef WAVELET_WAVELET_H
#define WAVELET_WAVELET_H

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

        KernelPair flipped() const;
    private:
        cv::Mat _lowpass;
        cv::Mat _highpass;
    };

    WaveletFilterBank(
        const KernelPair& synthesis_kernels,
        const KernelPair& analysis_kernels
    );
    WaveletFilterBank(
        const cv::Mat& synthesis_lowpass,
        const cv::Mat& synthesis_highpass,
        const cv::Mat& analysis_lowpass,
        const cv::Mat& analysis_highpass
    );
    WaveletFilterBank() = default;
    WaveletFilterBank(const WaveletFilterBank& other) = default;
    WaveletFilterBank(WaveletFilterBank&& other) = default;

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
    void inverse_stage1_lowpass(cv::InputArray data, cv::OutputArray output, int border_type=cv::BORDER_DEFAULT) const;

    cv::Mat inverse_stage1_highpass(cv::InputArray data, int border_type=cv::BORDER_DEFAULT) const;
    void inverse_stage1_highpass(cv::InputArray data, cv::OutputArray output, int border_type=cv::BORDER_DEFAULT) const;

    cv::Mat inverse_stage2_lowpass(cv::InputArray data, int border_type=cv::BORDER_DEFAULT) const;
    void inverse_stage2_lowpass(cv::InputArray data, cv::OutputArray output, int border_type=cv::BORDER_DEFAULT) const;

    cv::Mat inverse_stage2_highpass(cv::InputArray data, int border_type=cv::BORDER_DEFAULT) const;
    void inverse_stage2_highpass(cv::InputArray data, cv::OutputArray output, int border_type=cv::BORDER_DEFAULT) const;

    template <typename T>
    static KernelPair build_analysis_kernels(const std::vector<T>& coeffs);
    template <typename T>
    static KernelPair build_synthesis_kernels(const std::vector<T>& coeffs);

    template <typename T>
    static void negate_odds(std::vector<T>& coeffs);
    template <typename T>
    static void negate_evens(std::vector<T>& coeffs);

    KernelPair synthesis_kernels() const { return _synthesis_kernels.flipped(); }
    KernelPair analysis_kernels() const { return _analysis_kernels.flipped(); }

protected:
    void downsample_rows(cv::InputArray data, cv::OutputArray output) const;
    void downsample_cols(cv::InputArray data, cv::OutputArray output) const;
    void upsample_rows(cv::InputArray data, cv::OutputArray output) const;
    void upsample_cols(cv::InputArray data, cv::OutputArray output) const;
    void filter_rows(cv::InputArray data, cv::OutputArray output, const cv::Mat& kernel, int border_type) const;
    void filter_cols(cv::InputArray data, cv::OutputArray output, const cv::Mat& kernel, int border_type) const;

    KernelPair _synthesis_kernels;
    KernelPair _analysis_kernels;
};
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
        int order;
        int vanishing_moments_psi;
        int vanishing_moments_phi;
        int support_width;
        bool orthogonal;
        bool biorthogonal;
        Symmetry symmetry;
        bool compact_support;
        std::string family_name;
        std::string short_name;
        FilterBank filter_bank;
    };

public:
    Wavelet(
        int order,
        int vanishing_moments_psi,
        int vanishing_moments_phi,
        int support_width,
        bool orthogonal,
        bool biorthogonal,
        Symmetry symmetry,
        bool compact_support,
        const std::string& family_name,
        const std::string& short_name,
        const FilterBank& filter_bank
    );
    Wavelet() = delete;

    int order() const { return _p->order; }
    int vanishing_moments_psi() const { return _p->vanishing_moments_psi; }
    int vanishing_moments_phi() const { return _p->vanishing_moments_phi; }
    int support_width() const { return _p->support_width; }
    bool orthogonal() const { return _p->orthogonal; }
    bool biorthogonal() const { return _p->biorthogonal; }
    Symmetry symmetry() const { return _p->symmetry; }
    bool compact_support() const { return _p->compact_support; }
    std::string family_name() const { return _p->family_name; }
    std::string short_name() const { return _p->short_name; }
    const FilterBank& filter_bank() const { return _p->filter_bank; }

    static Wavelet create(const std::string& name);
    template<class... Args>
    static void register_factory(const std::string& name, Wavelet factory(Args...), const Args&... args);
    static std::vector<std::string> registered_wavelets();

private:
    std::shared_ptr<WaveletImpl> _p;
    static std::map<std::string, std::function<Wavelet()>> _wavelet_factories;
};


/**
 * Factories
*/
Wavelet daubechies(int order);
Wavelet haar();

}   // namespace wavelet

#endif

