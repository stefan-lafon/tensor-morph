#include "Advisor.h"
#include "codegen/MemoryAdvisor.h"
#include "codegen/ComputeAdvisor.h"
#include <stdexcept>

/**
 * Concrete implementation that routes calls to the transpiled models.
 */
class ModelAdvisor : public Advisor {
private:
    std::string profile_name;

public:
    explicit ModelAdvisor(const std::string& profile) : profile_name(profile) {}

    float Predict(const std::vector<float>& features) const override {
        // Route to the specific generated class based on profile.
        if (profile_name == "memory_bound") {
            return MemoryAdvisor::predict(features);
        } else if (profile_name == "compute_bound") {
            return ComputeAdvisor::predict(features);
        }

        throw std::runtime_error("Unknown hardware profile: " + profile_name);
    }

    std::string GetProfileName() const override {
        return profile_name;
    }
};

/**
 * Factory method to instantiate the correct advisor.
 */
std::unique_ptr<Advisor> CreateAdvisor(const std::string& profile) {
    return std::make_unique<ModelAdvisor>(profile);
}