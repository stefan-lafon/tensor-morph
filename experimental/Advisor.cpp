#include "experimental/Advisor.h"
#include "experimental/codegen/MemoryAdvisor.h"
#include "experimental/codegen/ComputeAdvisor.h"
#include <memory>
#include <string>

namespace mlir {
namespace tensormorph {

/**
 * Internal wrapper that selects the appropriate static prediction 
 * logic based on the requested hardware profile.
 */
class ModelAdvisor : public Advisor {
public:
    explicit ModelAdvisor(const std::string& profile) : profileName(profile) {}

    float Predict(const std::vector<float>& features) const override {
        if (profileName == "memory_bound") {
            return MemoryAdvisor::predict(features);
        } else if (profileName == "compute_bound") {
            return ComputeAdvisor::predict(features);
        }
        return 0.0f;
    }

    std::string GetProfileName() const override {
        return profileName;
    }

private:
    std::string profileName;
};

/**
 * Factory function to create the requested advisor profile.
 */
std::unique_ptr<Advisor> CreateAdvisor(const std::string& profile) {
    return std::make_unique<ModelAdvisor>(profile);
}

} // namespace tensormorph
} // namespace mlir